import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Prefix-Cached Attention"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        num_heads: int,
        cache_len: int,
        new_len: int,
        head_dim: int,
    ):
        total_len = cache_len + new_len
        assert Q.shape == (num_heads, new_len, head_dim)
        assert K.shape == (num_heads, total_len, head_dim)
        assert V.shape == (num_heads, total_len, head_dim)
        assert output.shape == (num_heads, new_len, head_dim)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32

        scale = 1.0 / math.sqrt(head_dim)

        # scores: (num_heads, new_len, total_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale

        # Causal mask: query token i (at absolute position cache_len+i) attends to
        # key token j iff j <= cache_len + i.
        # This gives full access to the KV cache and causal access within new tokens.
        i_idx = torch.arange(new_len, device=Q.device).unsqueeze(1)  # (new_len, 1)
        j_idx = torch.arange(total_len, device=Q.device).unsqueeze(0)  # (1, total_len)
        mask = j_idx <= cache_len + i_idx  # (new_len, total_len)

        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        output.copy_(torch.bmm(attn_weights, V))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_heads": (ctypes.c_int, "in"),
            "cache_len": (ctypes.c_int, "in"),
            "new_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, num_heads, cache_len, new_len, head_dim, zero_inputs=False):
        total_len = cache_len + new_len
        dtype = torch.float32
        device = self.device
        if zero_inputs:
            Q = torch.zeros(num_heads, new_len, head_dim, device=device, dtype=dtype)
            K = torch.zeros(num_heads, total_len, head_dim, device=device, dtype=dtype)
            V = torch.zeros(num_heads, total_len, head_dim, device=device, dtype=dtype)
        else:
            Q = torch.randn(num_heads, new_len, head_dim, device=device, dtype=dtype)
            K = torch.randn(num_heads, total_len, head_dim, device=device, dtype=dtype)
            V = torch.randn(num_heads, total_len, head_dim, device=device, dtype=dtype)
        output = torch.zeros(num_heads, new_len, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_heads": num_heads,
            "cache_len": cache_len,
            "new_len": new_len,
            "head_dim": head_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        num_heads = 2
        cache_len = 2
        new_len = 2
        head_dim = 4
        device = self.device
        dtype = torch.float32

        Q = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
            ],
            device=device,
            dtype=dtype,
        )
        K = torch.tensor(
            [
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                [
                    [0.0, 1.0, 0.0, -1.0],
                    [-1.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        V = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [-1.0, -2.0, -3.0, -4.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0],
                    [-2.0, -3.0, -4.0, -5.0],
                ],
            ],
            device=device,
            dtype=dtype,
        )
        output = torch.zeros(num_heads, new_len, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_heads": num_heads,
            "cache_len": cache_len,
            "new_len": new_len,
            "head_dim": head_dim,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: single decode step against a single cached token
        tests.append(self._make_test_case(1, 1, 1, 4))

        # Edge case: zero inputs
        tests.append(self._make_test_case(2, 2, 2, 4, zero_inputs=True))

        # cache_len=0: pure causal self-attention over new tokens
        tests.append(self._make_test_case(2, 0, 4, 8))

        # Single decode step (new_len=1) — typical autoregressive generation
        tests.append(self._make_test_case(4, 16, 1, 32))

        # Power-of-2 sizes
        tests.append(self._make_test_case(4, 32, 16, 32))

        # Larger power-of-2
        tests.append(self._make_test_case(8, 64, 32, 64))

        # Non-power-of-2 sizes
        tests.append(self._make_test_case(4, 30, 15, 32))

        # Non-power-of-2 with more heads
        tests.append(self._make_test_case(6, 100, 50, 32))

        # Long cache, short new chunk
        tests.append(self._make_test_case(8, 255, 3, 64))

        # Realistic dimensions (LLaMA-style), short chunk
        tests.append(self._make_test_case(16, 128, 64, 64))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # LLaMA-3 8B style: 32 heads, head_dim=128
        # cache_len=1024 (prior context), new_len=512 (chunk being prefilled)
        return self._make_test_case(32, 1024, 512, 128)
