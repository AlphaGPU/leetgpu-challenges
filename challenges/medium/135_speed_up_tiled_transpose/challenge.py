import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Speed Up Tiled Transpose"
    atol = 0.0
    rtol = 0.0
    num_gpus = 1
    access_tier = "premium"
    track = "kernel-interview"
    problem_type = "slow_kernel"
    bottleneck = "bank_conflicts"
    target_speedup = 1.5
    baseline_language = "cuda"
    hints = [
        "Both global loads and stores are already coalesced; the slowdown is inside shared memory.",
        "Shared memory has 32 banks of 4 bytes. What bank does tile[threadIdx.x][threadIdx.y] hit?",
        "Padding the tile to [32][33] or swizzling the index removes the conflicts.",
    ]

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
        assert input.shape == (rows, cols)
        assert output.shape == (cols, rows)
        assert input.dtype == torch.float32
        assert output.dtype == torch.float32
        output.copy_(input.t())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "rows": (ctypes.c_int, "in"),
            "cols": (ctypes.c_int, "in"),
        }

    def _make_test(self, rows: int, cols: int):
        f32 = torch.float32
        return {
            "input": torch.empty((rows, cols), device=self.device, dtype=f32).uniform_(-10.0, 10.0),
            "output": torch.empty((cols, rows), device=self.device, dtype=f32),
            "rows": rows,
            "cols": cols,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        f32 = torch.float32
        return {
            "input": torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device, dtype=f32
            ),
            "output": torch.empty((3, 2), device=self.device, dtype=f32),
            "rows": 2,
            "cols": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]
        tests.append(self._make_test(1, 1))
        tests.append(self._make_test(1, 50))
        tests.append(self._make_test(50, 1))
        zeros = self._make_test(4, 4)
        zeros["input"].zero_()
        tests.append(zeros)
        # 32 x 32 is exactly one tile; 33 x 31 exercises partial tiles on both axes
        tests.append(self._make_test(32, 32))
        tests.append(self._make_test(33, 31))
        tests.append(self._make_test(100, 255))
        tests.append(self._make_test(512, 1024))
        tests.append(self._make_test(3000, 2000))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test(8192, 8192)
