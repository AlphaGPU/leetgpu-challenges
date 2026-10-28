import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


def _pcg_hash(x: torch.Tensor) -> torch.Tensor:
    # PCG hash (Jarzynski & Olano, 2020) on uint32 values held in an int64 tensor
    state = (x * 747796405 + 2891336453) & 0xFFFFFFFF
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & 0xFFFFFFFF
    return (word >> 22) ^ word


class Challenge(ChallengeBase):
    name = "Top-p Sampling"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        logits: torch.Tensor,
        p: torch.Tensor,
        seed: torch.Tensor,
        sampled_token: torch.Tensor,
        vocab_size: int,
    ):
        assert logits.shape == (vocab_size,)
        assert p.shape == (1,)
        assert seed.shape == (1,)
        assert sampled_token.shape == (1,)
        assert logits.dtype == torch.float32
        assert p.dtype == torch.float32

        p_value = p.item()
        seed_value = seed.item()

        max_logit = torch.max(logits)
        exp_logits = torch.exp(logits - max_logit)
        probs = exp_logits / torch.sum(exp_logits)

        # stable: tokens with equal probability stay in increasing index order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, stable=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)

        cutoff_idx = torch.searchsorted(cumsum, p_value, right=False).item()
        cutoff_idx = min(cutoff_idx + 1, vocab_size)

        nucleus_probs = sorted_probs[:cutoff_idx]
        nucleus_indices = sorted_indices[:cutoff_idx]

        nucleus_probs = nucleus_probs / torch.sum(nucleus_probs)

        # One uniform per token, hashed from (seed, token id) so every language can reproduce it.
        seed_hash = _pcg_hash(torch.tensor(seed_value & 0xFFFFFFFF, device=logits.device))
        bits = _pcg_hash((seed_hash + nucleus_indices) & 0xFFFFFFFF)
        u = (bits.double() + 0.5) / 2**32

        # Exponential race: the argmax of prob / -ln(u) is a draw from the renormalized nucleus.
        sampled_idx = torch.argmax(nucleus_probs.double() / -torch.log(u)).item()
        sampled_token[0] = nucleus_indices[sampled_idx]

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "p": (ctypes.POINTER(ctypes.c_float), "in"),
            "seed": (ctypes.POINTER(ctypes.c_int32), "in"),
            "sampled_token": (ctypes.POINTER(ctypes.c_int32), "out"),
            "vocab_size": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        logits = torch.tensor([1.0, 2.0, 3.0, 0.5], device=self.device, dtype=torch.float32)
        p = torch.tensor([0.9], device=self.device, dtype=torch.float32)
        seed = torch.tensor([42], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)

        return {
            "logits": logits,
            "p": p,
            "seed": seed,
            "sampled_token": sampled_token,
            "vocab_size": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        logits = torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=torch.float32)
        p = torch.tensor([0.95], device=self.device, dtype=torch.float32)
        seed = torch.tensor([123], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 3,
            }
        )

        logits = torch.randn(10, device=self.device, dtype=torch.float32)
        p = torch.tensor([0.9], device=self.device, dtype=torch.float32)
        seed = torch.tensor([456], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 10,
            }
        )

        logits = torch.randn(100, device=self.device, dtype=torch.float32) * 5.0
        p = torch.tensor([0.85], device=self.device, dtype=torch.float32)
        seed = torch.tensor([789], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 100,
            }
        )

        logits = torch.zeros(50, device=self.device, dtype=torch.float32)
        logits[0] = 10.0
        p = torch.tensor([0.5], device=self.device, dtype=torch.float32)
        seed = torch.tensor([111], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 50,
            }
        )

        logits = torch.randn(500, device=self.device, dtype=torch.float32) * 3.0
        p = torch.tensor([0.92], device=self.device, dtype=torch.float32)
        seed = torch.tensor([222], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 500,
            }
        )

        logits = torch.linspace(-5, 5, 200, device=self.device, dtype=torch.float32)
        p = torch.tensor([0.8], device=self.device, dtype=torch.float32)
        seed = torch.tensor([333], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 200,
            }
        )

        logits = torch.randn(1000, device=self.device, dtype=torch.float32) * 2.0
        p = torch.tensor([0.95], device=self.device, dtype=torch.float32)
        seed = torch.tensor([444], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 1000,
            }
        )

        logits = torch.randn(5000, device=self.device, dtype=torch.float32)
        p = torch.tensor([0.9], device=self.device, dtype=torch.float32)
        seed = torch.tensor([555], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)
        tests.append(
            {
                "logits": logits,
                "p": p,
                "seed": seed,
                "sampled_token": sampled_token,
                "vocab_size": 5000,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        vocab_size = 50000
        logits = torch.randn(vocab_size, device=self.device, dtype=torch.float32) * 3.0
        p = torch.tensor([0.9], device=self.device, dtype=torch.float32)
        seed = torch.tensor([999], device=self.device, dtype=torch.int32)
        sampled_token = torch.zeros(1, device=self.device, dtype=torch.int32)

        return {
            "logits": logits,
            "p": p,
            "seed": seed,
            "sampled_token": sampled_token,
            "vocab_size": vocab_size,
        }
