import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Speed Up Histogram"
    atol = 0
    rtol = 0
    num_gpus = 1
    access_tier = "premium"
    listed = False
    track = "kernel-interview"
    problem_type = "slow_kernel"
    bottleneck = "atomic_contention"
    target_speedup = 6.0
    baseline_language = "cuda"
    hints = [
        "With only a few bins, millions of threads serialize on the same global addresses.",
        "Give each block a private histogram in shared memory and merge it once at the end.",
        "Warp-aggregated atomics (__match_any_sync) cut contention further.",
    ]

    def reference_impl(self, input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
        assert input.shape == (N,)
        assert histogram.shape == (num_bins,)
        assert input.dtype == torch.int32
        assert histogram.dtype == torch.int32
        counts = torch.bincount(input.long(), minlength=num_bins)
        histogram.copy_(counts.to(torch.int32))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "histogram": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "num_bins": (ctypes.c_int, "in"),
        }

    def _make_test(self, N: int, num_bins: int):
        return {
            "input": torch.randint(0, num_bins, (N,), device=self.device, dtype=torch.int32),
            "histogram": torch.empty((num_bins,), device=self.device, dtype=torch.int32),
            "N": N,
            "num_bins": num_bins,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return {
            "input": torch.tensor([0, 1, 2, 1, 0, 1], device=self.device, dtype=torch.int32),
            "histogram": torch.empty((4,), device=self.device, dtype=torch.int32),
            "N": 6,
            "num_bins": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]
        tests.append(self._make_test(1, 1))
        # every element in one bin (maximum contention); other bins must be 0
        hot = self._make_test(4096, 8)
        hot["input"].fill_(7)
        tests.append(hot)
        # all-zero input lands in bin 0
        zeros = self._make_test(100, 3)
        zeros["input"].zero_()
        tests.append(zeros)
        tests.append(self._make_test(3, 1024))
        tests.append(self._make_test(256, 32))
        tests.append(self._make_test(255, 100))
        tests.append(self._make_test(10000, 1024))
        tests.append(self._make_test(1000000, 32))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test(50000000, 32)
