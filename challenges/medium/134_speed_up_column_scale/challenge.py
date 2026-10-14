import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Speed Up Column Scale"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "premium"
    track = "kernel-interview"
    problem_type = "slow_kernel"
    bottleneck = "uncoalesced_loads"
    target_speedup = 4.0
    baseline_language = "cuda"
    hints = [
        "Look at which memory addresses 32 neighbouring threads of a warp touch.",
        "Map consecutive threadIdx.x to consecutive columns so each warp reads one contiguous row.",
        "Check global load efficiency / sectors per request in the profiler.",
    ]

    def reference_impl(
        self,
        input: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        output: torch.Tensor,
        M: int,
        N: int,
    ):
        assert input.shape == (M, N)
        assert gamma.shape == (N,)
        assert beta.shape == (N,)
        assert output.shape == (M, N)
        assert input.dtype == torch.float32
        assert gamma.dtype == torch.float32
        assert beta.dtype == torch.float32
        assert output.dtype == torch.float32
        output.copy_(input * gamma.unsqueeze(0) + beta.unsqueeze(0))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def _make_test(self, M: int, N: int, low: float = -10.0, high: float = 10.0):
        f32 = torch.float32
        return {
            "input": torch.empty((M, N), device=self.device, dtype=f32).uniform_(low, high),
            "gamma": torch.empty((N,), device=self.device, dtype=f32).uniform_(-2.0, 2.0),
            "beta": torch.empty((N,), device=self.device, dtype=f32).uniform_(-2.0, 2.0),
            "output": torch.empty((M, N), device=self.device, dtype=f32),
            "M": M,
            "N": N,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        f32 = torch.float32
        return {
            "input": torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device, dtype=f32
            ),
            "gamma": torch.tensor([2.0, 0.5, -1.0], device=self.device, dtype=f32),
            "beta": torch.tensor([1.0, 0.0, 3.0], device=self.device, dtype=f32),
            "output": torch.empty((2, 3), device=self.device, dtype=f32),
            "M": 2,
            "N": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]
        tests.append(self._make_test(1, 1))
        # single row / single column catch swapped M and N strides
        tests.append(self._make_test(1, 37))
        tests.append(self._make_test(37, 1))
        zeros = self._make_test(4, 4)
        zeros["input"].zero_()
        tests.append(zeros)
        # non-square, so using row instead of col to index gamma/beta is caught
        tests.append(self._make_test(30, 100))
        tests.append(self._make_test(64, 64))
        tests.append(self._make_test(255, 33, -100.0, 100.0))
        tests.append(self._make_test(512, 1024))
        tests.append(self._make_test(1000, 3000))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test(8192, 8192)
