import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Row-wise Argmax"
    atol = 0.0
    rtol = 0.0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, logits: torch.Tensor, output: torch.Tensor, B: int, V: int):
        assert logits.shape == (B, V), f"Expected logits.shape=({B}, {V}), got {logits.shape}"
        assert output.shape == (B,), f"Expected output.shape=({B},), got {output.shape}"
        assert logits.dtype == torch.float32
        assert output.dtype == torch.int32

        # ties resolve to the smallest column index
        row_max = logits.max(dim=1, keepdim=True).values
        cols = torch.arange(V, device=logits.device, dtype=torch.int64).expand(B, V)
        candidates = torch.where(logits == row_max, cols, torch.full_like(cols, V))
        output.copy_(candidates.min(dim=1).values.to(torch.int32))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_int), "out"),
            "B": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
        }

    def _make_test(self, B: int, V: int, low: float = -10.0, high: float = 10.0):
        logits = torch.empty((B, V), device=self.device, dtype=torch.float32).uniform_(low, high)
        output = torch.empty(B, device=self.device, dtype=torch.int32)
        return {"logits": logits, "output": output, "B": B, "V": V}

    def generate_example_test(self) -> Dict[str, Any]:
        logits = torch.tensor(
            [[1.0, 3.0, 2.0, -1.0], [-2.0, -0.5, -0.5, -3.0]],
            device=self.device,
            dtype=torch.float32,
        )
        output = torch.empty(2, device=self.device, dtype=torch.int32)
        return {"logits": logits, "output": output, "B": 2, "V": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        tests.append(self._make_test(1, 1))
        tests.append(self._make_test(3, 2))

        zeros = self._make_test(4, 16)
        zeros["logits"].zero_()
        tests.append(zeros)

        negative = self._make_test(8, 100, -100.0, -1.0)
        tests.append(negative)

        # maximum in the last column, then duplicated maxima
        last = self._make_test(16, 255)
        last["logits"][:, -1] = 50.0
        tests.append(last)

        ties = self._make_test(32, 1000)
        ties["logits"][:, 10] = 20.0
        ties["logits"][:, 900] = 20.0
        tests.append(ties)

        tests.append(self._make_test(64, 1024))
        tests.append(self._make_test(30, 32000))
        tests.append(self._make_test(128, 4096, -1.0, 1.0))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test(256, 128000)
