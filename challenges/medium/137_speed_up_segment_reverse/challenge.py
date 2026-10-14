import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Speed Up Segment Reverse"
    atol = 0.0
    rtol = 0.0
    num_gpus = 1
    access_tier = "premium"
    track = "kernel-interview"
    problem_type = "slow_kernel"
    bottleneck = "low_occupancy"
    target_speedup = 4.0
    baseline_language = "cuda"
    hints = [
        "How many warps fit on one SM with 32-thread blocks using up to 32 KB of shared memory?",
        "Check achieved occupancy: memory latency is not being hidden.",
        "Use more threads per block, or drop staging: out[i] = in[S - 1 - i] is coalesced.",
    ]

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, S: int):
        assert input.shape == (N,)
        assert output.shape == (N,)
        assert N % S == 0
        assert input.dtype == torch.float32
        assert output.dtype == torch.float32
        output.copy_(input.view(N // S, S).flip(1).reshape(N))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
        }

    def _make_test(self, segments: int, S: int):
        f32 = torch.float32
        N = segments * S
        return {
            "input": torch.empty((N,), device=self.device, dtype=f32).uniform_(-10.0, 10.0),
            "output": torch.empty((N,), device=self.device, dtype=f32),
            "N": N,
            "S": S,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        f32 = torch.float32
        return {
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.device, dtype=f32),
            "output": torch.empty((6,), device=self.device, dtype=f32),
            "N": 6,
            "S": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]
        # S = 1 is the identity; a single whole-array segment is a full reverse
        tests.append(self._make_test(5, 1))
        tests.append(self._make_test(1, 7))
        zeros = self._make_test(4, 4)
        zeros["input"].zero_()
        tests.append(zeros)
        # S = 2 catches reversing the whole array instead of each segment
        tests.append(self._make_test(64, 2))
        tests.append(self._make_test(30, 100))
        tests.append(self._make_test(16, 255))
        tests.append(self._make_test(8, 1024))
        # maximum segment length
        tests.append(self._make_test(4, 8192))
        tests.append(self._make_test(1000, 33))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test(8192, 8192)
