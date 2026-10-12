import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Shared-Prefix Cascade Attention"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K_prefix: torch.Tensor,
        V_prefix: torch.Tensor,
        K_suffix: torch.Tensor,
        V_suffix: torch.Tensor,
        output: torch.Tensor,
        batch_size: int,
        num_heads: int,
        prefix_len: int,
        suffix_len: int,
        head_dim: int,
    ):
        B, H, P, L, D = batch_size, num_heads, prefix_len, suffix_len, head_dim
        assert Q.shape == (B, H, L, D)
        assert K_prefix.shape == (H, P, D)
        assert V_prefix.shape == (H, P, D)
        assert K_suffix.shape == (B, H, L, D)
        assert V_suffix.shape == (B, H, L, D)
        assert output.shape == (B, H, L, D)
        for t in (Q, K_prefix, V_prefix, K_suffix, V_suffix, output):
            assert t.dtype == torch.float32

        scale = 1.0 / math.sqrt(D)
        # Every sequence in the batch sees the same prefix keys/values.
        K = torch.cat([K_prefix.unsqueeze(0).expand(B, H, P, D), K_suffix], dim=2)
        V = torch.cat([V_prefix.unsqueeze(0).expand(B, H, P, D), V_suffix], dim=2)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale  # (B, H, L, P + L)

        # Query i sees all P prefix keys and suffix keys 0..i (key index j <= P + i).
        q_idx = torch.arange(L, device=Q.device).unsqueeze(1)
        k_idx = torch.arange(P + L, device=Q.device).unsqueeze(0)
        mask = k_idx <= (q_idx + P)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        output.copy_(torch.matmul(attn, V))

    def get_solve_signature(self) -> Dict[str, tuple]:
        fp = ctypes.POINTER(ctypes.c_float)
        return {
            "Q": (fp, "in"),
            "K_prefix": (fp, "in"),
            "V_prefix": (fp, "in"),
            "K_suffix": (fp, "in"),
            "V_suffix": (fp, "in"),
            "output": (fp, "out"),
            "batch_size": (ctypes.c_int, "in"),
            "num_heads": (ctypes.c_int, "in"),
            "prefix_len": (ctypes.c_int, "in"),
            "suffix_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _make_test(self, B, H, P, L, D, kind="randn", scale=1.0) -> Dict[str, Any]:
        dev, dt = self.device, torch.float32

        def gen(*shape):
            if kind == "zeros":
                return torch.zeros(*shape, device=dev, dtype=dt)
            if kind == "uniform":
                return torch.empty(*shape, device=dev, dtype=dt).uniform_(-scale, scale)
            return torch.randn(*shape, device=dev, dtype=dt) * scale

        return {
            "Q": gen(B, H, L, D),
            "K_prefix": gen(H, P, D),
            "V_prefix": torch.randn(H, P, D, device=dev, dtype=dt),
            "K_suffix": gen(B, H, L, D),
            "V_suffix": torch.randn(B, H, L, D, device=dev, dtype=dt),
            "output": torch.zeros(B, H, L, D, device=dev, dtype=dt),
            "batch_size": B,
            "num_heads": H,
            "prefix_len": P,
            "suffix_len": L,
            "head_dim": D,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, dt = self.device, torch.float32
        Q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[1.0, 1.0], [0.0, 0.0]]]], device=dev)
        K_prefix = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=dev)
        V_prefix = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device=dev)
        K_suffix = torch.tensor(
            [[[[1.0, 1.0], [0.0, 0.0]]], [[[0.0, 1.0], [1.0, 0.0]]]], device=dev
        )
        V_suffix = torch.tensor(
            [[[[5.0, 6.0], [7.0, 8.0]]], [[[-1.0, -2.0], [0.0, 0.0]]]], device=dev
        )
        return {
            "Q": Q.to(dt),
            "K_prefix": K_prefix.to(dt),
            "V_prefix": V_prefix.to(dt),
            "K_suffix": K_suffix.to(dt),
            "V_suffix": V_suffix.to(dt),
            "output": torch.zeros(2, 1, 2, 2, device=dev, dtype=dt),
            "batch_size": 2,
            "num_heads": 1,
            "prefix_len": 2,
            "suffix_len": 2,
            "head_dim": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(0)
        tests = [self.generate_example_test()]
        # Single query per sequence (decode against a shared system prompt).
        tests.append(self._make_test(4, 2, 16, 1, 8))
        # No shared prefix: degenerates to plain causal attention over the suffix.
        tests.append(self._make_test(2, 2, 0, 5, 4))
        # Single prefix token, batch of one: catches off-by-one in the prefix/suffix boundary.
        tests.append(self._make_test(1, 1, 1, 3, 4))
        # Zero inputs: softmax must be uniform over exactly P + i + 1 visible keys.
        tests.append(self._make_test(3, 2, 7, 6, 16, kind="zeros"))
        # Large-magnitude logits: catches softmax without max-subtraction (overflow).
        tests.append(self._make_test(2, 4, 32, 32, 32, scale=6.0))
        # Negative/mixed uniform values with non-power-of-2 sizes and tile tails.
        tests.append(self._make_test(3, 3, 100, 30, 24, kind="uniform", scale=2.0))
        # Power-of-2 sizes; a prefix much longer than the suffix.
        tests.append(self._make_test(4, 4, 256, 64, 64))
        # Suffix longer than prefix: catches masks that treat the prefix as causal.
        tests.append(self._make_test(2, 2, 17, 128, 64))
        # Realistic: many sequences sharing one long prompt.
        tests.append(self._make_test(8, 8, 1024, 128, 128))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(42)
        return self._make_test(16, 32, 4096, 256, 128)
