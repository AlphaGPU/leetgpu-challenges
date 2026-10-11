import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Scatter Add"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        output: torch.Tensor,
        N: int,
        D: int,
        M: int,
    ):
        assert src.shape == (N, D), f"Expected src.shape=({N}, {D}), got {src.shape}"
        assert index.shape == (N,), f"Expected index.shape=({N},), got {index.shape}"
        assert output.shape == (M, D), f"Expected output.shape=({M}, {D}), got {output.shape}"
        assert src.dtype == torch.float32
        assert index.dtype == torch.int32
        assert output.dtype == torch.float32

        output.zero_()
        output.index_add_(0, index.long(), src)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "src": (ctypes.POINTER(ctypes.c_float), "in"),
            "index": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "M": (ctypes.c_int, "in"),
        }

    def _make_test(self, N: int, D: int, M: int, low: float = -1.0, high: float = 1.0):
        src = torch.empty((N, D), device=self.device, dtype=torch.float32).uniform_(low, high)
        index = torch.randint(0, M, (N,), device=self.device, dtype=torch.int32)
        output = torch.empty((M, D), device=self.device, dtype=torch.float32)
        return {"src": src, "index": index, "output": output, "N": N, "D": D, "M": M}

    def generate_example_test(self) -> Dict[str, Any]:
        src = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            device=self.device,
            dtype=torch.float32,
        )
        index = torch.tensor([0, 2, 0, 1], device=self.device, dtype=torch.int32)
        output = torch.empty((3, 2), device=self.device, dtype=torch.float32)
        return {"src": src, "index": index, "output": output, "N": 4, "D": 2, "M": 3}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        tests.append(
            {
                "src": torch.tensor([[2.5]], device=self.device, dtype=torch.float32),
                "index": torch.tensor([0], device=self.device, dtype=torch.int32),
                "output": torch.empty((1, 1), device=self.device, dtype=torch.float32),
                "N": 1,
                "D": 1,
                "M": 1,
            }
        )

        # rows that receive no source rows must be zero
        tests.append(
            {
                "src": torch.tensor(
                    [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], device=self.device, dtype=torch.float32
                ),
                "index": torch.tensor([3, 3], device=self.device, dtype=torch.int32),
                "output": torch.empty((4, 3), device=self.device, dtype=torch.float32),
                "N": 2,
                "D": 3,
                "M": 4,
            }
        )

        zeros = self._make_test(16, 4, 4)
        zeros["src"].zero_()
        tests.append(zeros)

        # every row lands on the same destination (maximum contention)
        hot = self._make_test(1024, 8, 16)
        hot["index"].fill_(5)
        tests.append(hot)

        tests.append(self._make_test(32, 32, 8, -10.0, 10.0))
        tests.append(self._make_test(100, 30, 7))
        tests.append(self._make_test(255, 64, 300))
        tests.append(self._make_test(512, 128, 64, -5.0, 5.0))
        tests.append(self._make_test(4096, 256, 1000))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test(1000000, 128, 50000)
