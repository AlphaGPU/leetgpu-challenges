import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Repetition Penalty Logit Processor"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
        B: int,
        V: int,
        T: int,
    ):
        assert logits.shape == (B, V)
        assert input_ids.shape == (B, T)
        assert logits.dtype == torch.float32
        assert input_ids.dtype == torch.int32

        score = torch.gather(logits, 1, input_ids.long())
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, input_ids.long(), score)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "inout"),
            "input_ids": (ctypes.POINTER(ctypes.c_int), "in"),
            "penalty": (ctypes.c_float, "in"),
            "B": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
            "T": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        B: int,
        V: int,
        T: int,
        penalty: float = 1.2,
        seed: int = 0,
        zero_logits: bool = False,
    ) -> Dict[str, Any]:
        device = self.device
        torch.manual_seed(seed)
        if zero_logits:
            logits = torch.zeros(B, V, device=device, dtype=torch.float32)
        else:
            logits = torch.empty(B, V, device=device, dtype=torch.float32).uniform_(-5.0, 5.0)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.int32, device=device)
        return {
            "logits": logits,
            "input_ids": input_ids,
            "penalty": penalty,
            "B": B,
            "V": V,
            "T": T,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        logits = torch.tensor([[2.0, -1.0, 0.5, -3.0, 1.0]], device=device, dtype=torch.float32)
        input_ids = torch.tensor([[0, 3, 3]], device=device, dtype=torch.int32)
        return {
            "logits": logits,
            "input_ids": input_ids,
            "penalty": 2.0,
            "B": 1,
            "V": 5,
            "T": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        device = self.device
        tests = []

        tests.append(
            {
                "logits": torch.tensor(
                    [[2.0, -1.0, 0.5, -3.0, 1.0]], device=device, dtype=torch.float32
                ),
                "input_ids": torch.tensor([[0, 3, 3]], device=device, dtype=torch.int32),
                "penalty": 2.0,
                "B": 1,
                "V": 5,
                "T": 3,
            }
        )

        tests.append(
            {
                "logits": torch.tensor(
                    [[-4.0, -2.0, 0.0, 2.0]], device=device, dtype=torch.float32
                ),
                "input_ids": torch.tensor([[1, 2]], device=device, dtype=torch.int32),
                "penalty": 1.5,
                "B": 1,
                "V": 4,
                "T": 2,
            }
        )

        tests.append(
            {
                "logits": torch.tensor(
                    [
                        [3.0, -3.0, 1.5, -1.5, 0.0, 4.0, -2.0, 2.0],
                        [-1.0, 2.0, 0.5, -0.5, 1.0, -2.0, 3.0, -3.0],
                    ],
                    device=device,
                    dtype=torch.float32,
                ),
                "input_ids": torch.tensor(
                    [[0, 5, 7, 7], [1, 6, 6, 4]], device=device, dtype=torch.int32
                ),
                "penalty": 1.3,
                "B": 2,
                "V": 8,
                "T": 4,
            }
        )

        tests.append(self._make_test_case(B=1, V=32, T=1, penalty=1.2, seed=10))
        tests.append(self._make_test_case(B=4, V=64, T=16, penalty=1.1, seed=11))
        tests.append(self._make_test_case(B=2, V=255, T=30, penalty=1.5, seed=12))
        tests.append(
            self._make_test_case(B=2, V=128, T=100, penalty=1.25, seed=13, zero_logits=True)
        )
        tests.append(self._make_test_case(B=8, V=1024, T=256, penalty=1.1, seed=14))
        tests.append(self._make_test_case(B=4, V=8192, T=512, penalty=1.3, seed=15))
        tests.append(self._make_test_case(B=2, V=32000, T=1024, penalty=1.15, seed=16))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(B=64, V=131072, T=4096, penalty=1.2, seed=42)
