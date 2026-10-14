import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Speed Up Row Sum"
    atol = 1e-03
    rtol = 1e-04
    num_gpus = 1
    access_tier = "premium"
    track = "kernel-interview"
    problem_type = "slow_kernel"
    bottleneck = "warp_divergence"
    target_speedup = 3.0
    baseline_language = "cuda"
    hints = [
        "In the tree loop, how many threads of each warp do useful work at step s?",
        "tid % (2 * s) == 0 leaves most lanes idle; use sequential addressing (tid < s) instead.",
        "Accumulate many elements per thread first, then finish with __shfl_down_sync.",
    ]

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, M: int, N: int):
        assert input.shape == (M, N)
        assert output.shape == (M,)
        assert input.dtype == torch.float32
        assert output.dtype == torch.float32
        output.copy_(input.sum(dim=1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def _make_test(self, M: int, N: int, low: float = -1.0, high: float = 1.0):
        f32 = torch.float32
        return {
            "input": torch.empty((M, N), device=self.device, dtype=f32).uniform_(low, high),
            "output": torch.empty((M,), device=self.device, dtype=f32),
            "M": M,
            "N": N,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        f32 = torch.float32
        return {
            "input": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.5, -3.0]], device=self.device, dtype=f32
            ),
            "output": torch.empty((2,), device=self.device, dtype=f32),
            "M": 2,
            "N": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]
        tests.append(self._make_test(1, 1))
        tests.append(self._make_test(3, 2))
        zeros = self._make_test(4, 16)
        zeros["input"].zero_()
        tests.append(zeros)
        # cancellation: +x and -x pairs must sum to (near) zero
        cancel = self._make_test(8, 512, -10.0, 10.0)
        cancel["input"][:, 256:] = -cancel["input"][:, :256]
        tests.append(cancel)
        # N = 256 / 257 hit the block-size boundary and the first tail element
        tests.append(self._make_test(16, 256))
        tests.append(self._make_test(16, 257))
        tests.append(self._make_test(100, 255))
        tests.append(self._make_test(1024, 1000))
        tests.append(self._make_test(64, 8192))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test(16384, 4096)
