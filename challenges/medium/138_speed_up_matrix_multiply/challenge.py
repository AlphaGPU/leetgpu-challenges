import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Speed Up Matrix Multiply"
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "premium"
    track = "kernel-interview"
    problem_type = "slow_kernel"
    bottleneck = "missing_tiling"
    target_speedup = 2.5
    baseline_language = "cuda"
    hints = [
        "Each element of A and B is re-read from global memory by many threads.",
        "Compute arithmetic intensity: the naive kernel does 2 FLOPs per 8 bytes loaded.",
        "Stage tiles of A and B in shared memory, then compute several outputs per thread.",
    ]

    def reference_impl(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int
    ):
        assert A.shape == (M, K)
        assert B.shape == (K, N)
        assert C.shape == (M, N)
        assert A.dtype == torch.float32
        assert B.dtype == torch.float32
        assert C.dtype == torch.float32
        C.copy_(torch.matmul(A, B))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def _make_test(self, M: int, N: int, K: int):
        f32 = torch.float32
        return {
            "A": torch.empty((M, K), device=self.device, dtype=f32).uniform_(-1.0, 1.0),
            "B": torch.empty((K, N), device=self.device, dtype=f32).uniform_(-1.0, 1.0),
            "C": torch.empty((M, N), device=self.device, dtype=f32),
            "M": M,
            "N": N,
            "K": K,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        f32 = torch.float32
        return {
            "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device, dtype=f32),
            "B": torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], device=self.device, dtype=f32),
            "C": torch.empty((2, 3), device=self.device, dtype=f32),
            "M": 2,
            "N": 3,
            "K": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]
        tests.append(self._make_test(1, 1, 1))
        # K = 1 is an outer product; M = 1 / N = 1 catch swapped strides
        tests.append(self._make_test(4, 3, 1))
        tests.append(self._make_test(1, 64, 33))
        tests.append(self._make_test(64, 1, 33))
        zeros = self._make_test(16, 16, 16)
        zeros["A"].zero_()
        tests.append(zeros)
        tests.append(self._make_test(32, 32, 32))
        # non-multiples of common tile sizes on every axis
        tests.append(self._make_test(100, 30, 255))
        tests.append(self._make_test(255, 129, 67))
        tests.append(self._make_test(512, 512, 1024))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test(4096, 4096, 4096)
