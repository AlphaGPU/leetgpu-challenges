import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

# The six ways to keep 2 of 4 positions, as (idx0, idx1) with idx0 < idx1.
_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


class Challenge(ChallengeBase):
    name = "Semi-Structured Sparse MatMul"
    # Each output is a dot product of K/2 nonzero terms with operands in [-1, 1]. Against a float64
    # recomputation, float32 results in any summation order (blocked, split-K, atomics) differ by
    # <= 5e-5 for K <= 16,384 (~1.3e-5 on the performance case), well inside atol + rtol * |ref|.
    # Rounding x or the weights to FP16 / BF16 misses by ~1e-2 on 75% of outputs, and dropping
    # or misplacing a single nonzero term misses by orders of magnitude more.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        w_values: torch.Tensor,
        w_meta: torch.Tensor,
        y: torch.Tensor,
        M: int,
        N: int,
        K: int,
    ):
        assert K % 4 == 0, f"K must be divisible by 4, got {K}"
        G = K // 4
        assert x.shape == (M, K), f"Expected x.shape=({M}, {K}), got {x.shape}"
        assert w_values.shape == (N, K // 2), f"Expected w_values.shape=({N}, {K // 2})"
        assert w_meta.shape == (N, G), f"Expected w_meta.shape=({N}, {G}), got {w_meta.shape}"
        assert y.shape == (M, N), f"Expected y.shape=({M}, {N}), got {y.shape}"
        assert x.dtype == torch.float32
        assert w_values.dtype == torch.float32
        assert w_meta.dtype == torch.uint8
        assert y.dtype == torch.float32

        # Decode the 2-bit in-group positions: bits 1:0 -> idx0, bits 3:2 -> idx1.
        idx0 = (w_meta & 3).to(torch.int64)
        idx1 = ((w_meta >> 2) & 3).to(torch.int64)
        idx = torch.stack([idx0, idx1], dim=-1)  # [N, G, 2]

        # Scatter the two kept values of every group into a dense [N, G, 4] weight.
        w_dense = torch.zeros((N, G, 4), device=x.device, dtype=torch.float32)
        w_dense.scatter_(2, idx, w_values.reshape(N, G, 2))
        y.copy_(torch.matmul(x, w_dense.reshape(N, K).t()))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "w_values": (ctypes.POINTER(ctypes.c_float), "in"),
            "w_meta": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "y": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def _pack(self, pair_ids: torch.Tensor) -> torch.Tensor:
        pairs = torch.tensor(_PAIRS, device=self.device, dtype=torch.int32)
        chosen = pairs[pair_ids.to(torch.int64)]  # [..., 2]
        return (chosen[..., 0] | (chosen[..., 1] << 2)).to(torch.uint8)

    def _case(
        self,
        M: int,
        N: int,
        K: int,
        seed: int,
        x_range=(-1.0, 1.0),
        w_range=(-1.0, 1.0),
        pair_ids: torch.Tensor = None,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        x = torch.empty((M, K), device=dev, dtype=f32).uniform_(*x_range)
        w_values = torch.empty((N, K // 2), device=dev, dtype=f32).uniform_(*w_range)
        if pair_ids is None:
            pair_ids = torch.randint(0, 6, (N, K // 4), device=dev)
        return {
            "x": x,
            "w_values": w_values,
            "w_meta": self._pack(pair_ids),
            # Pre-filled with garbage: y is not zero-initialized, so every element must be
            # written and accumulating (split-K / atomicAdd) solutions must initialize it first.
            "y": torch.full((M, N), 123.0, device=dev, dtype=f32),
            "M": M,
            "N": N,
            "K": K,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1.0, 0.0, -1.0, 2.0, 0.0, 1.0, -2.0, 1.0]],
            device=dev,
            dtype=f32,
        )
        # Row 0: groups keep (0,1) and (2,3) -> dense [1, 1, 0, 0, 0, 0, 1, 1]
        # Row 1: groups keep (0,3) and (1,2) -> dense [2, 0, 0, -1, 0, 0.5, 3, 0]
        # Row 2: groups keep (1,3) and (0,2) -> dense [0, -1, 0, 1, 1, 0, -2, 0]
        w_values = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 0.5, 3.0], [-1.0, 1.0, 1.0, -2.0]],
            device=dev,
            dtype=f32,
        )
        w_meta = torch.tensor([[4, 14], [12, 9], [13, 8]], device=dev, dtype=torch.uint8)
        return {
            "x": x,
            "w_values": w_values,
            "w_meta": w_meta,
            "y": torch.full((2, 3), 123.0, device=dev, dtype=f32),
            "M": 2,
            "N": 3,
            "K": 8,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dev = self.device
        tests = []

        # Hand-computed example: y = [[18, 22, -7], [0, -5.5, 6]]. Uses all six position pairs
        # and checks the reference implementation against independently computed values.
        tests.append(self.generate_example_test())

        # Minimal: a single group (K = 4), one token, one output feature.
        tests.append(self._case(1, 1, 4, seed=1))

        # Every position pair in every row and group slot, with distinct x columns and distinct
        # values: row n uses pair (n + g) % 6 in group g. Catches swapping the idx0/idx1 bit
        # fields (values land in each other's column), reading meta as a 4-bit keep-mask,
        # assuming idx1 = idx0 + 1, and reusing group 0's pattern for the whole row.
        n_ids = torch.arange(6, device=dev).view(6, 1)
        g_ids = torch.arange(12, device=dev).view(1, 12)
        tests.append(self._case(3, 6, 48, seed=2, pair_ids=(n_ids + g_ids) % 6))

        # Zero activations: y must be exactly 0 (catches unwritten / un-cleared outputs).
        zero = self._case(4, 64, 64, seed=3)
        zero["x"].zero_()
        tests.append(zero)

        # All-negative x and weights: every term is positive, so skipping the trailing groups
        # of a row (K / 4 = 257 is odd and prime, a tail for any 2/4/8-group vectorized loop)
        # produces a strictly smaller result instead of cancelling out.
        tests.append(self._case(2, 33, 1028, seed=4, x_range=(-1.0, -0.1), w_range=(-1.0, -0.1)))

        # Non-square M != N != K with tails in every dimension: catches computing x @ W instead
        # of x @ W^T, swapping the row strides of w_values (K/2) and w_meta (K/4), and missing
        # bounds checks on N and M tiles.
        tests.append(self._case(5, 37, 132, seed=5))

        # Power-of-2 sizes.
        tests.append(self._case(16, 256, 512, seed=6))
        tests.append(self._case(32, 128, 1024, seed=7))

        # Non-power-of-2 sizes.
        tests.append(self._case(7, 255, 300, seed=8))
        tests.append(self._case(100, 30, 20, seed=9))

        # Large token count (M at its upper bound): catches decode kernels that hard-code a
        # small number of rows of x per block / in registers.
        tests.append(self._case(256, 48, 256, seed=10))

        # Long reduction (K at its upper bound): a split-K solution that overwrites instead of
        # accumulating partial sums, or drops the last K chunk, fails here.
        tests.append(self._case(8, 512, 8192, seed=11))

        # Realistic decode step: 16 sequences through a 4,096 x 4,096 projection.
        tests.append(self._case(16, 4096, 4096, seed=12))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Batched decode (16 sequences) through the fused gate/up projection of a Llama-3-8B
        # MLP (2 x 14,336 output features, hidden size 4,096) pruned to 2:4 sparsity.
        return self._case(16, 28672, 4096, seed=159)
