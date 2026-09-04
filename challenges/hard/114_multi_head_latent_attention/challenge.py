import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Multi-Head Latent Attention Decode"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        W_UK: torch.Tensor,
        W_UV: torch.Tensor,
        output: torch.Tensor,
        num_heads: int,
        seq_len: int,
        kv_lora_rank: int,
        head_dim: int,
        rope_dim: int,
    ):
        assert q.shape == (num_heads, head_dim + rope_dim)
        assert kv_cache.shape == (seq_len, kv_lora_rank + rope_dim)
        assert W_UK.shape == (num_heads, head_dim, kv_lora_rank)
        assert W_UV.shape == (num_heads, kv_lora_rank, head_dim)
        assert output.shape == (num_heads, head_dim)
        assert q.dtype == kv_cache.dtype == W_UK.dtype == W_UV.dtype == output.dtype

        # Split the query into its content ("nope") and rotary ("pe") parts, and the
        # cache into the compressed latent vectors and the shared rotary keys.
        q_nope = q[:, :head_dim]  # (num_heads, head_dim)
        q_pe = q[:, head_dim:]  # (num_heads, rope_dim)
        c_kv = kv_cache[:, :kv_lora_rank]  # (seq_len, kv_lora_rank)
        k_pe = kv_cache[:, kv_lora_rank:]  # (seq_len, rope_dim)

        scale = 1.0 / math.sqrt(head_dim + rope_dim)

        # Weight absorption: push W_UK into the query so keys never leave latent space.
        q_latent = torch.bmm(q_nope.unsqueeze(1), W_UK).squeeze(1)  # (num_heads, kv_lora_rank)

        # Scores against every cached position: latent dot product + decoupled rotary term.
        scores = (q_latent @ c_kv.transpose(0, 1) + q_pe @ k_pe.transpose(0, 1)) * scale
        attn = torch.softmax(scores, dim=-1)  # (num_heads, seq_len)

        # Attend in latent space, then up-project with W_UV.
        latent_out = attn @ c_kv  # (num_heads, kv_lora_rank)
        output.copy_(torch.bmm(latent_out.unsqueeze(1), W_UV).squeeze(1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "q": (ctypes.POINTER(ctypes.c_float), "in"),
            "kv_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_UK": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_UV": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_heads": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "kv_lora_rank": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
            "rope_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        num_heads,
        seq_len,
        kv_lora_rank,
        head_dim,
        rope_dim,
        zero_q=False,
        zero_cache=False,
        seed=None,
    ):
        dtype = torch.float32
        device = self.device
        if seed is not None:
            torch.manual_seed(seed)

        if zero_q:
            q = torch.zeros(num_heads, head_dim + rope_dim, device=device, dtype=dtype)
        else:
            q = torch.randn(num_heads, head_dim + rope_dim, device=device, dtype=dtype)

        if zero_cache:
            kv_cache = torch.zeros(seq_len, kv_lora_rank + rope_dim, device=device, dtype=dtype)
        else:
            kv_cache = torch.randn(seq_len, kv_lora_rank + rope_dim, device=device, dtype=dtype)

        W_UK = torch.randn(
            num_heads, head_dim, kv_lora_rank, device=device, dtype=dtype
        ) / math.sqrt(head_dim)
        W_UV = torch.randn(
            num_heads, kv_lora_rank, head_dim, device=device, dtype=dtype
        ) / math.sqrt(kv_lora_rank)
        output = torch.empty(num_heads, head_dim, device=device, dtype=dtype)
        return {
            "q": q,
            "kv_cache": kv_cache,
            "W_UK": W_UK,
            "W_UV": W_UV,
            "output": output,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "kv_lora_rank": kv_lora_rank,
            "head_dim": head_dim,
            "rope_dim": rope_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        num_heads, seq_len, kv_lora_rank, head_dim, rope_dim = 2, 2, 2, 2, 1

        q = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        kv_cache = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=device, dtype=dtype)
        W_UK = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], device=device, dtype=dtype
        )
        W_UV = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]], device=device, dtype=dtype
        )
        output = torch.empty(num_heads, head_dim, device=device, dtype=dtype)
        return {
            "q": q,
            "kv_cache": kv_cache,
            "W_UK": W_UK,
            "W_UV": W_UV,
            "output": output,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "kv_lora_rank": kv_lora_rank,
            "head_dim": head_dim,
            "rope_dim": rope_dim,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge case: a single cached position, softmax degenerates to 1.0
        tests.append(self._make_test_case(1, 1, 2, 2, 1, seed=0))

        # Edge case: two heads, two cached positions
        tests.append(self._make_test_case(2, 2, 4, 2, 2, seed=1))

        # Edge case: zero query -> uniform attention over the cache
        tests.append(self._make_test_case(1, 3, 8, 4, 2, zero_q=True, seed=2))

        # Edge case: zero cache -> zero output
        tests.append(self._make_test_case(4, 4, 8, 4, 4, zero_cache=True, seed=3))

        # Power-of-2 shapes
        tests.append(self._make_test_case(8, 64, 64, 32, 16, seed=4))
        tests.append(self._make_test_case(16, 256, 128, 64, 32, seed=5))

        # Non-power-of-2 shapes
        tests.append(self._make_test_case(3, 30, 48, 24, 12, seed=6))
        tests.append(self._make_test_case(5, 100, 96, 40, 20, seed=7))
        tests.append(self._make_test_case(6, 255, 64, 32, 16, seed=8))

        # Realistic decode step: DeepSeek-style rank 512 latent cache
        tests.append(self._make_test_case(32, 1024, 512, 128, 64, seed=9))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # DeepSeek-V3 decode step: 128 heads, rank-512 latent cache, 4K context
        return self._make_test_case(128, 4096, 512, 128, 64, seed=42)
