import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Split-KV Attention Reduction"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        partial_out: torch.Tensor,
        partial_lse: torch.Tensor,
        output: torch.Tensor,
        num_splits: int,
        num_heads: int,
        head_dim: int,
    ):
        assert partial_out.shape == (num_splits, num_heads, head_dim)
        assert partial_lse.shape == (num_splits, num_heads)
        assert output.shape == (num_heads, head_dim)
        assert partial_out.dtype == torch.float32
        assert partial_lse.dtype == torch.float32
        assert output.dtype == torch.float32

        lse_max = partial_lse.amax(dim=0, keepdim=True)  # [1, num_heads]
        weights = torch.exp(partial_lse - lse_max)  # [num_splits, num_heads]
        weights_sum = weights.sum(dim=0, keepdim=True)  # [1, num_heads]
        weights_norm = weights / weights_sum  # [num_splits, num_heads]

        merged = (weights_norm.unsqueeze(-1) * partial_out).sum(dim=0)  # [num_heads, head_dim]
        output.copy_(merged)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "partial_out": (ctypes.POINTER(ctypes.c_float), "in"),
            "partial_lse": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_splits": (ctypes.c_int, "in"),
            "num_heads": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case_from_attention(
        self,
        num_splits: int,
        num_heads: int,
        seq_per_split: int,
        head_dim: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Build a realistic (partial_out, partial_lse) pair by actually splitting a
        scaled dot-product attention over a fake KV cache."""
        device = self.device
        torch.manual_seed(seed)

        total_seq = num_splits * seq_per_split
        Q = torch.randn(num_heads, head_dim, device=device, dtype=torch.float32)
        K = torch.randn(num_heads, total_seq, head_dim, device=device, dtype=torch.float32)
        V = torch.randn(num_heads, total_seq, head_dim, device=device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(head_dim)
        partial_out = torch.empty(
            num_splits, num_heads, head_dim, device=device, dtype=torch.float32
        )
        partial_lse = torch.empty(num_splits, num_heads, device=device, dtype=torch.float32)

        for s in range(num_splits):
            start = s * seq_per_split
            end = start + seq_per_split
            K_s = K[:, start:end, :]  # [H, chunk, D]
            V_s = V[:, start:end, :]  # [H, chunk, D]
            scores = (Q.unsqueeze(1) @ K_s.transpose(-2, -1)).squeeze(1) * scale  # [H, chunk]
            score_max = scores.amax(dim=-1, keepdim=True)  # [H, 1]
            exp_scores = torch.exp(scores - score_max)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)  # [H, 1]
            lse = (score_max + torch.log(sum_exp)).squeeze(-1)  # [H]
            probs = exp_scores / sum_exp  # [H, chunk]
            out = (probs.unsqueeze(-1) * V_s).sum(dim=1)  # [H, D]
            partial_out[s] = out
            partial_lse[s] = lse

        output = torch.empty(num_heads, head_dim, device=device, dtype=torch.float32)
        return {
            "partial_out": partial_out,
            "partial_lse": partial_lse,
            "output": output,
            "num_splits": num_splits,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }

    def _make_test_case_synthetic(
        self,
        num_splits: int,
        num_heads: int,
        head_dim: int,
        seed: int,
        lse_std: float = 2.0,
        out_std: float = 1.0,
    ) -> Dict[str, Any]:
        """Build a (partial_out, partial_lse) pair from random tensors, without simulating
        the underlying attention. Faster than _make_test_case_from_attention for large sizes."""
        device = self.device
        torch.manual_seed(seed)
        partial_out = (
            torch.randn(num_splits, num_heads, head_dim, device=device, dtype=torch.float32)
            * out_std
        )
        partial_lse = (
            torch.randn(num_splits, num_heads, device=device, dtype=torch.float32) * lse_std
        )
        output = torch.empty(num_heads, head_dim, device=device, dtype=torch.float32)
        return {
            "partial_out": partial_out,
            "partial_lse": partial_lse,
            "output": output,
            "num_splits": num_splits,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        num_splits, num_heads, head_dim = 2, 2, 2
        # Head 0: both splits have equal log-sum-exp -> simple 50/50 average.
        # Head 1: split 0 has lse = ln 3, split 1 has lse = 0 -> 3:1 weighting -> 0.75/0.25.
        partial_out = torch.tensor(
            [
                # split 0: [head 0, head 1]
                [[1.0, 2.0], [4.0, 0.0]],
                # split 1: [head 0, head 1]
                [[3.0, 4.0], [0.0, 8.0]],
            ],
            device=device,
            dtype=dtype,
        )
        partial_lse = torch.tensor(
            [
                [0.0, math.log(3.0)],
                [0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
        output = torch.empty(num_heads, head_dim, device=device, dtype=dtype)
        return {
            "partial_out": partial_out,
            "partial_lse": partial_lse,
            "output": output,
            "num_splits": num_splits,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests: List[Dict[str, Any]] = []

        # Edge: single split (output must equal partial_out[0]).
        tests.append(self._make_test_case_synthetic(1, 4, 8, seed=1))

        # Edge: two splits, one head.
        tests.append(self._make_test_case_synthetic(2, 1, 16, seed=2))

        # Edge: three splits, small dims.
        tests.append(self._make_test_case_from_attention(3, 2, 4, 8, seed=3))

        # Zero partial_out with varied lse -> output must be zero.
        zero_case = self._make_test_case_synthetic(4, 4, 8, seed=4)
        zero_case["partial_out"].zero_()
        tests.append(zero_case)

        # All-equal lse -> uniform average.
        eq_case = self._make_test_case_synthetic(4, 8, 16, seed=5)
        eq_case["partial_lse"].fill_(2.5)
        tests.append(eq_case)

        # Power-of-two: 8 splits, 16 heads, head_dim 64.
        tests.append(self._make_test_case_from_attention(8, 16, 32, 64, seed=6))

        # Non-power-of-two splits, mid-sized.
        tests.append(self._make_test_case_synthetic(17, 12, 64, seed=7, lse_std=3.0))

        # Non-power-of-two head_dim.
        tests.append(self._make_test_case_synthetic(16, 30, 96, seed=8, lse_std=1.5))

        # Realistic: 32 splits (16K KV split into 512-chunks), 32 heads, head_dim 128.
        tests.append(self._make_test_case_from_attention(32, 32, 32, 128, seed=9))

        # Larger realistic: 64 splits, 16 heads, head_dim 128.
        tests.append(self._make_test_case_synthetic(64, 16, 128, seed=10, lse_std=2.5))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Flash-Decoding for a very long KV cache: 128 splits, 64 heads, head_dim 128.
        return self._make_test_case_synthetic(
            num_splits=128, num_heads=64, head_dim=128, seed=42, lse_std=2.0
        )
