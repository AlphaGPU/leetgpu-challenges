import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Beam Search Step",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        beam_scores: torch.Tensor,
        token_logprobs: torch.Tensor,
        new_beam_scores: torch.Tensor,
        parent_beam_indices: torch.Tensor,
        next_tokens: torch.Tensor,
        B: int,
        K: int,
        V: int,
    ):
        assert beam_scores.shape == (B, K)
        assert token_logprobs.shape == (B, K, V)
        assert new_beam_scores.shape == (B, K)
        assert parent_beam_indices.shape == (B, K)
        assert next_tokens.shape == (B, K)
        assert beam_scores.dtype == torch.float32
        assert token_logprobs.dtype == torch.float32
        assert new_beam_scores.dtype == torch.float32
        assert parent_beam_indices.dtype == torch.int32
        assert next_tokens.dtype == torch.int32
        assert beam_scores.device.type == "cuda"
        assert token_logprobs.device.type == "cuda"
        assert new_beam_scores.device.type == "cuda"
        assert parent_beam_indices.device.type == "cuda"
        assert next_tokens.device.type == "cuda"

        candidates = beam_scores.unsqueeze(-1) + token_logprobs
        flat = candidates.view(B, K * V)
        top_vals, top_idx = torch.topk(flat, K, dim=-1, sorted=True)
        new_beam_scores.copy_(top_vals)
        parent_beam_indices.copy_((top_idx // V).to(torch.int32))
        next_tokens.copy_((top_idx % V).to(torch.int32))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "beam_scores": (ctypes.POINTER(ctypes.c_float), "in"),
            "token_logprobs": (ctypes.POINTER(ctypes.c_float), "in"),
            "new_beam_scores": (ctypes.POINTER(ctypes.c_float), "out"),
            "parent_beam_indices": (ctypes.POINTER(ctypes.c_int), "out"),
            "next_tokens": (ctypes.POINTER(ctypes.c_int), "out"),
            "B": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        B: int,
        K: int,
        V: int,
        seed: int,
        beam_scale: float = 1.0,
        logprob_scale: float = 2.0,
        zero_beam_scores: bool = False,
    ) -> Dict[str, Any]:
        device = "cuda"
        torch.manual_seed(seed)
        if zero_beam_scores:
            beam_scores = torch.zeros((B, K), device=device, dtype=torch.float32)
        else:
            beam_scores = torch.randn((B, K), device=device, dtype=torch.float32) * beam_scale
        token_logprobs = torch.randn((B, K, V), device=device, dtype=torch.float32) * logprob_scale
        return {
            "beam_scores": beam_scores,
            "token_logprobs": token_logprobs,
            "new_beam_scores": torch.zeros((B, K), device=device, dtype=torch.float32),
            "parent_beam_indices": torch.zeros((B, K), device=device, dtype=torch.int32),
            "next_tokens": torch.zeros((B, K), device=device, dtype=torch.int32),
            "B": B,
            "K": K,
            "V": V,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        beam_scores = torch.tensor(
            [[-0.5, -1.0]],
            device=device,
            dtype=torch.float32,
        )
        token_logprobs = torch.tensor(
            [
                [
                    [-0.3, -1.2, -2.0, -2.5],
                    [-0.4, -0.1, -1.6, -3.2],
                ]
            ],
            device=device,
            dtype=torch.float32,
        )
        return {
            "beam_scores": beam_scores,
            "token_logprobs": token_logprobs,
            "new_beam_scores": torch.zeros((1, 2), device=device, dtype=torch.float32),
            "parent_beam_indices": torch.zeros((1, 2), device=device, dtype=torch.int32),
            "next_tokens": torch.zeros((1, 2), device=device, dtype=torch.int32),
            "B": 1,
            "K": 2,
            "V": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        # Example values
        tests.append(self.generate_example_test())
        # Single beam (K=1) — degenerate to argmax over each row of token_logprobs
        tests.append(self._make_test_case(B=2, K=1, V=8, seed=1))
        # Smallest non-trivial case
        tests.append(self._make_test_case(B=3, K=2, V=2, seed=2))
        # Power-of-2 K and V
        tests.append(self._make_test_case(B=2, K=4, V=8, seed=3))
        tests.append(self._make_test_case(B=4, K=8, V=128, seed=4))
        # Non-power-of-2 V
        tests.append(self._make_test_case(B=8, K=4, V=255, seed=5))
        tests.append(self._make_test_case(B=2, K=8, V=1003, seed=6))
        # All-zero beam scores: pure top-K over token_logprobs
        tests.append(self._make_test_case(B=4, K=4, V=64, seed=7, zero_beam_scores=True))
        # Realistic vocab sizes
        tests.append(self._make_test_case(B=4, K=4, V=10000, seed=8))
        tests.append(self._make_test_case(B=8, K=8, V=4096, seed=9))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(B=16, K=8, V=50000, seed=2024)
