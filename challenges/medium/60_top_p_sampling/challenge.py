import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Top-p Sampling", atol=1e-05, rtol=1e-05, num_gpus=1, access_tier="free"
        )

    def reference_impl(
        self,
        logits: torch.Tensor,
        p: torch.Tensor,
        top_p_probs: torch.Tensor,
        vocab_size: int,
    ):
        assert logits.shape == (vocab_size,)
        assert p.shape == (1,)
        assert top_p_probs.shape == (vocab_size,)
        assert logits.dtype == torch.float32
        assert p.dtype == torch.float32
        assert top_p_probs.dtype == torch.float32

        p_value = p.item()

        max_logit = torch.max(logits)
        exp_logits = torch.exp(logits - max_logit)
        probs = exp_logits / torch.sum(exp_logits)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)

        cutoff_idx = torch.searchsorted(cumsum, p_value, right=False).item()
        cutoff_idx = min(cutoff_idx + 1, vocab_size)

        nucleus_probs = sorted_probs[:cutoff_idx]
        nucleus_indices = sorted_indices[:cutoff_idx]
        nucleus_probs = nucleus_probs / torch.sum(nucleus_probs)

        top_p_probs.zero_()
        top_p_probs[nucleus_indices] = nucleus_probs

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "p": (ctypes.POINTER(ctypes.c_float), "in"),
            "top_p_probs": (ctypes.POINTER(ctypes.c_float), "out"),
            "vocab_size": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        logits = torch.tensor([1.0, 2.0, 3.0, 0.5], device="cuda", dtype=torch.float32)
        p = torch.tensor([0.9], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(4, device="cuda", dtype=torch.float32)

        return {
            "logits": logits,
            "p": p,
            "top_p_probs": top_p_probs,
            "vocab_size": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        logits = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
        p = torch.tensor([0.95], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(3, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "top_p_probs": top_p_probs,
                "vocab_size": 3,
            }
        )

        logits = torch.randn(10, device="cuda", dtype=torch.float32)
        p = torch.tensor([0.9], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(10, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "top_p_probs": top_p_probs,
                "vocab_size": 10,
            }
        )

        logits = torch.randn(100, device="cuda", dtype=torch.float32) * 5.0
        p = torch.tensor([0.85], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(100, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "top_p_probs": top_p_probs,
                "vocab_size": 100,
            }
        )

        logits = torch.zeros(50, device="cuda", dtype=torch.float32)
        logits[0] = 10.0
        p = torch.tensor([0.5], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(50, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "top_p_probs": top_p_probs,
                "vocab_size": 50,
            }
        )

        logits = torch.randn(500, device="cuda", dtype=torch.float32) * 3.0
        p = torch.tensor([0.92], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(500, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "top_p_probs": top_p_probs,
                "vocab_size": 500,
            }
        )

        logits = torch.linspace(-5, 5, 200, device="cuda", dtype=torch.float32)
        p = torch.tensor([0.8], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(200, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "top_p_probs": top_p_probs,
                "vocab_size": 200,
            }
        )

        logits = torch.randn(1000, device="cuda", dtype=torch.float32) * 2.0
        p = torch.tensor([0.95], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(1000, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "top_p_probs": top_p_probs,
                "vocab_size": 1000,
            }
        )

        logits = torch.randn(5000, device="cuda", dtype=torch.float32)
        p = torch.tensor([0.9], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(5000, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "top_p_probs": top_p_probs,
                "vocab_size": 5000,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        vocab_size = 50000
        logits = torch.randn(vocab_size, device="cuda", dtype=torch.float32) * 3.0
        p = torch.tensor([0.9], device="cuda", dtype=torch.float32)
        top_p_probs = torch.zeros(vocab_size, device="cuda", dtype=torch.float32)

        return {
            "logits": logits,
            "p": p,
            "top_p_probs": top_p_probs,
            "vocab_size": vocab_size,
        }
