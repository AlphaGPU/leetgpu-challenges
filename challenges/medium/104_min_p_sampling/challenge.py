import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Min-P Sampling"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        min_p: float,
        B: int,
        V: int,
    ):
        assert logits.shape == (B, V)
        assert probs.shape == (B, V)
        assert logits.dtype == torch.float32
        assert probs.dtype == torch.float32

        max_logit, _ = torch.max(logits, dim=-1, keepdim=True)
        exp_shifted = torch.exp(logits - max_logit)
        keep = exp_shifted >= min_p
        masked = torch.where(keep, exp_shifted, torch.zeros_like(exp_shifted))
        denom = torch.sum(masked, dim=-1, keepdim=True)
        probs.copy_(masked / denom)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "probs": (ctypes.POINTER(ctypes.c_float), "out"),
            "min_p": (ctypes.c_float, "in"),
            "B": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        B, V = 2, 4
        logits = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, -2.0]],
            device=self.device,
            dtype=dtype,
        )
        probs = torch.empty(B, V, device=self.device, dtype=dtype)
        return {
            "logits": logits,
            "probs": probs,
            "min_p": 0.1,
            "B": B,
            "V": V,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # single row, tiny vocab
        B, V = 1, 3
        tests.append(
            {
                "logits": torch.tensor([[1.0, 2.0, 3.0]], device=self.device, dtype=dtype),
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.05,
                "B": B,
                "V": V,
            }
        )

        # tied maxima (multiple winners survive)
        B, V = 1, 4
        tests.append(
            {
                "logits": torch.tensor([[3.0, 3.0, 1.0, -2.0]], device=self.device, dtype=dtype),
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.5,
                "B": B,
                "V": V,
            }
        )

        # min_p = 0 (no filtering => plain softmax)
        B, V = 2, 5
        tests.append(
            {
                "logits": torch.tensor(
                    [[-1.0, 0.0, 1.0, 2.0, 3.0], [0.5, 0.5, 0.5, 0.5, 0.5]],
                    device=self.device,
                    dtype=dtype,
                ),
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.0,
                "B": B,
                "V": V,
            }
        )

        # min_p near 1 (only the maximum survives)
        B, V = 3, 6
        tests.append(
            {
                "logits": torch.tensor(
                    [
                        [0.0, 1.0, 5.0, 2.0, 3.0, -1.0],
                        [10.0, -10.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.99,
                "B": B,
                "V": V,
            }
        )

        # all-zero logits (uniform distribution survives)
        B, V = 2, 8
        tests.append(
            {
                "logits": torch.zeros(B, V, device=self.device, dtype=dtype),
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.1,
                "B": B,
                "V": V,
            }
        )

        # power-of-two vocab with mixed positives/negatives
        B, V = 4, 128
        torch.manual_seed(0)
        tests.append(
            {
                "logits": torch.randn(B, V, device=self.device, dtype=dtype) * 2.0,
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.1,
                "B": B,
                "V": V,
            }
        )

        # non-power-of-two vocab
        B, V = 8, 255
        torch.manual_seed(1)
        tests.append(
            {
                "logits": torch.randn(B, V, device=self.device, dtype=dtype) * 3.0,
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.05,
                "B": B,
                "V": V,
            }
        )

        # peaked distribution (a few dominating tokens)
        B, V = 4, 1024
        torch.manual_seed(2)
        logits = torch.full((B, V), -5.0, device=self.device, dtype=dtype)
        logits[0, 17] = 10.0
        logits[0, 99] = 9.5
        logits[1, 3] = 8.0
        logits[2, 500] = 7.0
        logits[3, 1000] = 12.0
        logits[3, 0] = 11.0
        tests.append(
            {
                "logits": logits,
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.05,
                "B": B,
                "V": V,
            }
        )

        # realistic batch x vocab
        B, V = 16, 32000
        torch.manual_seed(3)
        tests.append(
            {
                "logits": torch.randn(B, V, device=self.device, dtype=dtype) * 2.5,
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.1,
                "B": B,
                "V": V,
            }
        )

        # larger batch, smaller vocab
        B, V = 64, 4096
        torch.manual_seed(4)
        tests.append(
            {
                "logits": torch.randn(B, V, device=self.device, dtype=dtype) * 1.5,
                "probs": torch.empty(B, V, device=self.device, dtype=dtype),
                "min_p": 0.02,
                "B": B,
                "V": V,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        B, V = 64, 128000
        torch.manual_seed(42)
        return {
            "logits": torch.randn(B, V, device=self.device, dtype=dtype) * 2.0,
            "probs": torch.empty(B, V, device=self.device, dtype=dtype),
            "min_p": 0.05,
            "B": B,
            "V": V,
        }
