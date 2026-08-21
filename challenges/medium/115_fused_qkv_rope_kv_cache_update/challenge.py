import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused QKV Projection with RoPE and KV Cache Update"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        W_qkv: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        Q_out: torch.Tensor,
        B: int,
        d_model: int,
        H_q: int,
        H_kv: int,
        D: int,
        S_max: int,
    ):
        assert x.shape == (B, d_model)
        assert W_qkv.shape == (d_model, (H_q + 2 * H_kv) * D)
        assert cos_sin_cache.shape == (S_max, D)
        assert positions.shape == (B,)
        assert K_cache.shape == (B, H_kv, S_max, D)
        assert V_cache.shape == (B, H_kv, S_max, D)
        assert Q_out.shape == (B, H_q, D)
        assert (
            x.dtype
            == W_qkv.dtype
            == cos_sin_cache.dtype
            == K_cache.dtype
            == V_cache.dtype
            == Q_out.dtype
            == torch.float32
        )
        assert positions.dtype == torch.int32

        qkv = x @ W_qkv  # [B, (H_q + 2 * H_kv) * D]
        q = qkv[:, : H_q * D].reshape(B, H_q, D)
        k = qkv[:, H_q * D : (H_q + H_kv) * D].reshape(B, H_kv, D)
        v = qkv[:, (H_q + H_kv) * D :].reshape(B, H_kv, D)

        pos = positions.long()
        cos_sin = cos_sin_cache[pos]  # [B, D]
        cos = cos_sin[:, : D // 2].unsqueeze(1)  # [B, 1, D // 2]
        sin = cos_sin[:, D // 2 :].unsqueeze(1)  # [B, 1, D // 2]

        def rope(t: torch.Tensor) -> torch.Tensor:
            t1 = t[..., : D // 2]
            t2 = t[..., D // 2 :]
            return torch.cat((t1 * cos - t2 * sin, t2 * cos + t1 * sin), dim=-1)

        Q_out.copy_(rope(q))

        b_idx = torch.arange(B, device=positions.device).unsqueeze(1).expand(B, H_kv)
        h_idx = torch.arange(H_kv, device=positions.device).unsqueeze(0).expand(B, H_kv)
        p_idx = pos.unsqueeze(1).expand(B, H_kv)
        K_cache[b_idx, h_idx, p_idx] = rope(k)
        V_cache[b_idx, h_idx, p_idx] = v

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_qkv": (ctypes.POINTER(ctypes.c_float), "in"),
            "cos_sin_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "positions": (ctypes.POINTER(ctypes.c_int), "in"),
            "K_cache": (ctypes.POINTER(ctypes.c_float), "inout"),
            "V_cache": (ctypes.POINTER(ctypes.c_float), "inout"),
            "Q_out": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "H_q": (ctypes.c_int, "in"),
            "H_kv": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "S_max": (ctypes.c_int, "in"),
        }

    def _rope_table(self, S_max: int, D: int) -> torch.Tensor:
        theta = 10000.0
        exponent = torch.arange(0, D, 2, device=self.device, dtype=torch.float32) / D
        inv_freq = 1.0 / (theta**exponent)  # [D // 2]
        t = torch.arange(S_max, device=self.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [S_max, D // 2]
        return torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1)  # [S_max, D]

    def _make_test_case(
        self,
        B: int,
        d_model: int,
        H_q: int,
        H_kv: int,
        D: int,
        S_max: int,
        seed: int,
        zero_x: bool = False,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        device = self.device
        dtype = torch.float32

        if zero_x:
            x = torch.zeros(B, d_model, device=device, dtype=dtype)
        else:
            x = torch.randn(B, d_model, device=device, dtype=dtype)
        W_qkv = torch.randn(d_model, (H_q + 2 * H_kv) * D, device=device, dtype=dtype) * 0.02
        cos_sin_cache = self._rope_table(S_max, D)
        positions = torch.randint(0, S_max, (B,), device=device, dtype=torch.int32)
        K_cache = torch.randn(B, H_kv, S_max, D, device=device, dtype=dtype) * 0.1
        V_cache = torch.randn(B, H_kv, S_max, D, device=device, dtype=dtype) * 0.1
        Q_out = torch.empty(B, H_q, D, device=device, dtype=dtype)

        return {
            "x": x,
            "W_qkv": W_qkv,
            "cos_sin_cache": cos_sin_cache,
            "positions": positions,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "Q_out": Q_out,
            "B": B,
            "d_model": d_model,
            "H_q": H_q,
            "H_kv": H_kv,
            "D": D,
            "S_max": S_max,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        B, d_model, H_q, H_kv, D, S_max = 1, 2, 1, 1, 2, 2

        x = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        W_qkv = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 2.0, 2.0],
            ],
            device=device,
            dtype=dtype,
        )
        cos_sin_cache = torch.tensor(
            [[1.0, 0.0], [0.5403023, 0.8414710]], device=device, dtype=dtype
        )
        positions = torch.tensor([1], device=device, dtype=torch.int32)
        K_cache = torch.zeros(B, H_kv, S_max, D, device=device, dtype=dtype)
        V_cache = torch.zeros(B, H_kv, S_max, D, device=device, dtype=dtype)
        Q_out = torch.empty(B, H_q, D, device=device, dtype=dtype)

        return {
            "x": x,
            "W_qkv": W_qkv,
            "cos_sin_cache": cos_sin_cache,
            "positions": positions,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "Q_out": Q_out,
            "B": B,
            "d_model": d_model,
            "H_q": H_q,
            "H_kv": H_kv,
            "D": D,
            "S_max": S_max,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge: single sequence, single head, minimum head dim, one cache slot
        tests.append(self._make_test_case(1, 4, 1, 1, 2, 1, seed=0))
        # Edge: two sequences, grouped-query with 2 queries per KV head
        tests.append(self._make_test_case(2, 4, 2, 1, 4, 2, seed=1))
        # Edge: three sequences, multi-head attention (H_q == H_kv)
        tests.append(self._make_test_case(3, 6, 3, 3, 2, 4, seed=2))
        # Zero input: projection is all zeros, cache slots overwritten with zeros
        tests.append(self._make_test_case(4, 8, 4, 2, 8, 8, seed=3, zero_x=True))
        # Power-of-2 sizes
        tests.append(self._make_test_case(16, 64, 8, 2, 16, 32, seed=4))
        tests.append(self._make_test_case(32, 128, 8, 8, 32, 64, seed=5))
        # Non-power-of-2 sizes
        tests.append(self._make_test_case(30, 100, 5, 5, 6, 30, seed=6))
        tests.append(self._make_test_case(100, 96, 6, 3, 10, 255, seed=7))
        # Realistic decode batches
        tests.append(self._make_test_case(64, 512, 8, 2, 64, 512, seed=8))
        tests.append(self._make_test_case(128, 1024, 16, 4, 64, 1024, seed=9))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # LLaMA-3 8B style decode step: 32 query heads, 8 KV heads, head_dim 128.
        return self._make_test_case(64, 4096, 32, 8, 128, 2048, seed=42)
