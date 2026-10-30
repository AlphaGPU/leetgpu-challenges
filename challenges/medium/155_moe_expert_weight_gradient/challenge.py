import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "MoE Expert Weight Gradient"
    # Each dW element is a sum of n_g products of values in [-1, 1] (n_g <= 4,720 here), so its
    # magnitude is O(sqrt(n_g) / 3) (max |dW| ~115 on the performance case). Against a float64
    # recomputation the float32 result differs by <= 5e-5; other float32 summation orders
    # (tiling, split-K, atomics) stay under 1% of the grader's atol + rtol * max|ref| budget.
    # Rounding X and dY to TF32 / FP16 exceeds it by ~2.6x and BF16 by ~20x. Attributing one row
    # to the wrong expert, dropping a boundary row, or leaving garbage in an empty expert misses
    # by orders of magnitude more.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        X: torch.Tensor,
        dY: torch.Tensor,
        group_offsets: torch.Tensor,
        dW: torch.Tensor,
        G: int,
        M_total: int,
        K: int,
        N: int,
    ):
        assert X.shape == (M_total, K), f"Expected X.shape=({M_total}, {K}), got {X.shape}"
        assert dY.shape == (M_total, N), f"Expected dY.shape=({M_total}, {N}), got {dY.shape}"
        assert group_offsets.shape == (
            G + 1,
        ), f"Expected group_offsets.shape=({G + 1},), got {group_offsets.shape}"
        assert dW.shape == (G, K, N), f"Expected dW.shape=({G}, {K}, {N}), got {dW.shape}"
        assert X.dtype == torch.float32
        assert dY.dtype == torch.float32
        assert dW.dtype == torch.float32
        assert group_offsets.dtype == torch.int32

        out = torch.zeros((G, K, N), device=X.device, dtype=torch.float32)
        offsets = group_offsets.tolist()
        for g in range(G):
            start, end = offsets[g], offsets[g + 1]
            if end > start:
                out[g] = torch.matmul(X[start:end].t(), dY[start:end])
        dW.copy_(out)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "dY": (ctypes.POINTER(ctypes.c_float), "in"),
            "group_offsets": (ctypes.POINTER(ctypes.c_int), "in"),
            "dW": (ctypes.POINTER(ctypes.c_float), "out"),
            "G": (ctypes.c_int, "in"),
            "M_total": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def _build(
        self,
        group_sizes: List[int],
        K: int,
        N: int,
        seed: int = 0,
        x_range=(-1.0, 1.0),
        dy_range=(-1.0, 1.0),
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        G = len(group_sizes)
        M_total = sum(group_sizes)
        X = torch.empty((M_total, K), device=dev, dtype=f32).uniform_(*x_range)
        dY = torch.empty((M_total, N), device=dev, dtype=f32).uniform_(*dy_range)
        offsets = [0]
        for s in group_sizes:
            offsets.append(offsets[-1] + s)
        return {
            "X": X,
            "dY": dY,
            "group_offsets": torch.tensor(offsets, device=dev, dtype=torch.int32),
            # Pre-filled with garbage (as in other backward challenges): dW is documented as not
            # zero-initialized, so empty experts must be written as zeros and accumulating
            # (e.g. atomicAdd split-K) solutions must initialize dW first.
            "dW": torch.full((G, K, N), 123.0, device=dev, dtype=f32),
            "G": G,
            "M_total": M_total,
            "K": K,
            "N": N,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "X": torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 4.0]], device=dev, dtype=f32),
            "dY": torch.tensor([[1.0, 0.0], [2.0, 1.0], [-1.0, 3.0]], device=dev, dtype=f32),
            "group_offsets": torch.tensor([0, 2, 2, 3], device=dev, dtype=torch.int32),
            "dW": torch.full((3, 2, 2), 123.0, device=dev, dtype=f32),
            "G": 3,
            "M_total": 3,
            "K": 2,
            "N": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Hand-computed example: dW[0] = [[7, 3], [0, -1]], dW[1] = 0 (empty expert),
        # dW[2] = [[0, 0], [-4, 12]]. Checks the reference against independent values.
        tests.append(self.generate_example_test())

        # Minimal: one expert, one token, K = N = 1 (dW is a single product).
        tests.append(self._build([1], K=1, N=1, seed=1))

        # One expert, K != N (3 x 5): catches computing dY^T X or swapping the K/N strides.
        tests.append(self._build([6], K=3, N=5, seed=2))

        # Every token routed to one expert, all others empty (including the first and last):
        # catches indexing experts by position among non-empty groups (result lands in dW[0]
        # instead of dW[2]) and writing nonzero values into empty experts.
        tests.append(self._build([0, 0, 37, 0], K=8, N=12, seed=3))

        # Zero activations: dW must be exactly 0 everywhere (catches unwritten output).
        zero = self._build([3, 5, 2], K=16, N=16, seed=4)
        zero["X"].zero_()
        tests.append(zero)

        # Single-token adjacent experts with all-positive X and all-negative dY: every term
        # has the same sign, so an off-by-one boundary (inclusive end, using offsets[g + 1]
        # as start) moves a whole term of magnitude >= 0.01 to the wrong expert.
        tests.append(
            self._build([1] * 8, K=4, N=4, seed=5, x_range=(0.1, 1.0), dy_range=(-1.0, -0.1))
        )

        # Power-of-2: 4 balanced experts.
        tests.append(self._build([16, 16, 16, 16], K=32, N=64, seed=6))

        # Non-power-of-2 sizes and dims: exercises K/N tile tails and token-chunk tails.
        tests.append(self._build([7, 11, 13], K=33, N=27, seed=7))

        # One hot expert with a long reduction plus tiny/empty ones: a split-K solution that
        # overwrites instead of accumulating partial sums, or drops the last partial chunk,
        # fails on expert 0; expert 2 is empty and must still be zeroed.
        tests.append(self._build([2049, 3, 0, 5], K=64, N=48, seed=8))

        # Many experts (64) with small, uneven counts including zeros: catches per-expert
        # indexing errors in dW (stride K * N) when G is large.
        torch.manual_seed(9)
        sizes = torch.randint(0, 20, (64,)).tolist()
        sizes[0], sizes[63] = 0, 0
        tests.append(self._build(sizes, K=24, N=40, seed=10))

        # Realistic: 8 experts, ~1-2K tokens each, hidden 256, expert intermediate 128.
        torch.manual_seed(11)
        sizes = torch.randint(1000, 2000, (8,)).tolist()
        tests.append(self._build(sizes, K=256, N=128, seed=12))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Backward pass of a fine-grained MoE expert projection: 4,096 tokens x top-8 routing
        # = 32,768 routed rows over 64 experts, hidden size 1,024, expert intermediate 512.
        # Expert loads follow a Zipf-like skew (hottest expert 4,720 rows, coldest non-empty 170)
        # and two experts receive no tokens, as in real, imperfectly balanced routing.
        G, M_total, K, N = 64, 32768, 1024, 512
        gen = torch.Generator().manual_seed(155)
        weights = 1.0 / torch.arange(1, G + 1, dtype=torch.float64) ** 0.8
        weights = weights[torch.randperm(G, generator=gen)]
        empty = torch.randperm(G, generator=gen)[:2]
        weights[empty] = 0.0
        sizes = torch.floor(weights / weights.sum() * M_total).long()
        sizes[int(torch.argmax(sizes))] += M_total - int(sizes.sum())
        return self._build(sizes.tolist(), K=K, N=N, seed=42)
