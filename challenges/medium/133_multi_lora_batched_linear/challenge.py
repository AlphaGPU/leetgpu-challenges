import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Multi-LoRA Batched Linear"
    # Outputs are O(1) dot products over K <= 4,096 (weights scaled by 1/sqrt(K));
    # float32 accumulation-order error stays near 1e-5, while using the wrong
    # adapter, applying an adapter to an id of -1 or forgetting the scale is off by >> 1e-3.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        adapter_ids: torch.Tensor,
        y: torch.Tensor,
        T: int,
        K: int,
        N: int,
        R: int,
        L: int,
        scale: float,
    ):
        assert x.shape == (T, K)
        assert W.shape == (K, N)
        assert A.shape == (L, K, R)
        assert B.shape == (L, R, N)
        assert adapter_ids.shape == (T,)
        assert y.shape == (T, N)
        assert x.dtype == W.dtype == A.dtype == B.dtype == y.dtype == torch.float32
        assert adapter_ids.dtype == torch.int32

        out = x @ W
        ids = adapter_ids.long()
        has = ids >= 0
        safe = ids.clamp(min=0)
        shrink = torch.bmm(x.unsqueeze(1), A[safe]).squeeze(1)  # (T, R)
        expand = torch.bmm(shrink.unsqueeze(1), B[safe]).squeeze(1)  # (T, N)
        out = out + scale * expand * has.unsqueeze(1).to(out.dtype)
        y.copy_(out)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "W": (ctypes.POINTER(ctypes.c_float), "in"),
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "adapter_ids": (ctypes.POINTER(ctypes.c_int), "in"),
            "y": (ctypes.POINTER(ctypes.c_float), "out"),
            "T": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "R": (ctypes.c_int, "in"),
            "L": (ctypes.c_int, "in"),
            "scale": (ctypes.c_float, "in"),
        }

    def _make(self, T, K, N, R, L, scale=2.0, seed=0, ids=None):
        torch.manual_seed(seed)

        def rnd(*shape, s=1.0):
            return torch.empty(*shape, device=self.device, dtype=torch.float32).uniform_(-s, s)

        if ids is None:
            adapter_ids = torch.randint(-1, L, (T,), device=self.device, dtype=torch.int32)
        else:
            adapter_ids = torch.tensor(ids, device=self.device, dtype=torch.int32)
        return {
            "x": rnd(T, K),
            "W": rnd(K, N, s=1.0 / K**0.5),
            "A": rnd(L, K, R, s=1.0 / K**0.5),
            "B": rnd(L, R, N, s=1.0 / R**0.5),
            "adapter_ids": adapter_ids,
            "y": torch.zeros(T, N, device=self.device, dtype=torch.float32),
            "T": T,
            "K": K,
            "N": N,
            "R": R,
            "L": L,
            "scale": scale,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "x": torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]], device=dev, dtype=f32),
            "W": torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=dev, dtype=f32),
            "A": torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]], device=dev, dtype=f32),
            "B": torch.tensor([[[1.0, 1.0]], [[1.0, -1.0]]], device=dev, dtype=f32),
            "adapter_ids": torch.tensor([0, 1, -1], device=dev, dtype=torch.int32),
            "y": torch.zeros(3, 2, device=dev, dtype=f32),
            "T": 3,
            "K": 2,
            "N": 2,
            "R": 1,
            "L": 2,
            "scale": 0.5,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]

        # Minimal: one token, one adapter, rank 1.
        tests.append(self._make(1, 1, 1, 1, 1, seed=1, ids=[0]))

        # Every token has id -1: output must equal the base projection exactly.
        tests.append(self._make(4, 8, 8, 4, 3, seed=2, ids=[-1, -1, -1, -1]))

        # All tokens on the last adapter: catches off-by-one adapter indexing.
        tests.append(self._make(5, 16, 12, 8, 4, seed=3, ids=[3, 3, 3, 3, 3]))

        # Zero input: output must be all zeros.
        z = self._make(3, 4, 4, 2, 2, seed=4)
        z["x"].zero_()
        tests.append(z)

        # Negative-only inputs, negative scale.
        neg = self._make(16, 32, 32, 8, 4, scale=-1.5, seed=5)
        neg["x"].abs_().neg_()
        tests.append(neg)

        # Interleaved adapters (no sorting by adapter), non-power-of-two shapes.
        tests.append(self._make(37, 100, 75, 12, 5, seed=6, ids=[i % 6 - 1 for i in range(37)]))

        # Power-of-two shapes, random mix including base-only tokens.
        tests.append(self._make(64, 256, 128, 16, 8, seed=7))

        # Realistic serving batch with many adapters.
        tests.append(self._make(256, 1024, 1024, 16, 16, seed=8))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make(1024, 4096, 4096, 16, 8, seed=0)
