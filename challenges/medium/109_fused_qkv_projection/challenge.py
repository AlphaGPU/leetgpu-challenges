import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "Fused QKV Projection"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        W_qkv: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        M: int,
        num_heads: int,
        head_dim: int,
    ):
        D = num_heads * head_dim
        assert x.shape == (M, D)
        assert W_qkv.shape == (3 * D, D)
        assert Q.shape == (num_heads, M, head_dim)
        assert K.shape == (num_heads, M, head_dim)
        assert V.shape == (num_heads, M, head_dim)
        assert x.dtype == W_qkv.dtype == Q.dtype == K.dtype == V.dtype == torch.float32

        # Fused projection: single matmul producing packed Q, K, V.
        qkv = x @ W_qkv.t()  # [M, 3*D]

        # Split into Q, K, V along the last dimension.
        q_flat, k_flat, v_flat = qkv.split(D, dim=-1)  # each [M, D]

        # Reshape to (M, num_heads, head_dim) and transpose to (num_heads, M, head_dim).
        Q.copy_(q_flat.reshape(M, num_heads, head_dim).transpose(0, 1))
        K.copy_(k_flat.reshape(M, num_heads, head_dim).transpose(0, 1))
        V.copy_(v_flat.reshape(M, num_heads, head_dim).transpose(0, 1))

    def reference_impl_jax(self, x, W_qkv, M, num_heads, head_dim):
        import jax.numpy as jnp

        D = num_heads * head_dim
        qkv = x @ W_qkv.T  # [M, 3*D]
        q_flat = qkv[:, :D]
        k_flat = qkv[:, D : 2 * D]
        v_flat = qkv[:, 2 * D :]

        def to_heads(t):
            return jnp.transpose(jnp.reshape(t, (M, num_heads, head_dim)), (1, 0, 2))

        return to_heads(q_flat), to_heads(k_flat), to_heads(v_flat)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_qkv": (ctypes.POINTER(ctypes.c_float), "in"),
            "Q": (ctypes.POINTER(ctypes.c_float), "out"),
            "K": (ctypes.POINTER(ctypes.c_float), "out"),
            "V": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "num_heads": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, M, num_heads, head_dim, zero_x=False):
        device = self.device
        dtype = torch.float32
        D = num_heads * head_dim
        if zero_x:
            x = torch.zeros(M, D, device=device, dtype=dtype)
        else:
            x = torch.randn(M, D, device=device, dtype=dtype) * 0.1
        W_qkv = torch.randn(3 * D, D, device=device, dtype=dtype) * 0.02
        Q = torch.empty(num_heads, M, head_dim, device=device, dtype=dtype)
        K = torch.empty(num_heads, M, head_dim, device=device, dtype=dtype)
        V = torch.empty(num_heads, M, head_dim, device=device, dtype=dtype)
        return {
            "x": x,
            "W_qkv": W_qkv,
            "Q": Q,
            "K": K,
            "V": V,
            "M": M,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        # M = 2 tokens, D = 4, num_heads = 2, head_dim = 2.
        M, num_heads, head_dim = 2, 2, 2

        # Input rows: two basis-like vectors.
        x = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        # W_qkv is [3*D=12, D=4]. Rows 0..3 are Q weights, 4..7 are K, 8..11 are V.
        # We build each 4x4 block to give a small, hand-checkable result.
        w_q = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        w_k = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
        w_v = torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
            ],
            device=device,
            dtype=dtype,
        )
        W_qkv = torch.cat([w_q, w_k, w_v], dim=0)  # [12, 4]

        Q = torch.empty(num_heads, M, head_dim, device=device, dtype=dtype)
        K = torch.empty(num_heads, M, head_dim, device=device, dtype=dtype)
        V = torch.empty(num_heads, M, head_dim, device=device, dtype=dtype)
        return {
            "x": x,
            "W_qkv": W_qkv,
            "Q": Q,
            "K": K,
            "V": V,
            "M": M,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge: single token, single head, tiny head_dim.
        tests.append(self._make_test_case(1, 1, 4))

        # Zero input.
        tests.append(self._make_test_case(4, 2, 8, zero_x=True))

        # Power-of-2 sizes, small.
        tests.append(self._make_test_case(16, 4, 8))

        # Power-of-2 sizes, moderate.
        tests.append(self._make_test_case(64, 8, 16))

        # Non-power-of-2 M.
        tests.append(self._make_test_case(30, 4, 16))

        # Non-power-of-2 M, larger head layout.
        tests.append(self._make_test_case(100, 8, 32))

        # Non-power-of-2 M, medium size.
        tests.append(self._make_test_case(255, 4, 32))

        # Negative-only inputs to exercise sign handling.
        M, num_heads, head_dim = 32, 4, 16
        D = num_heads * head_dim
        tests.append(
            {
                "x": torch.full((M, D), -0.5, device=self.device, dtype=torch.float32),
                "W_qkv": torch.randn(3 * D, D, device=self.device, dtype=torch.float32) * 0.02,
                "Q": torch.empty(num_heads, M, head_dim, device=self.device, dtype=torch.float32),
                "K": torch.empty(num_heads, M, head_dim, device=self.device, dtype=torch.float32),
                "V": torch.empty(num_heads, M, head_dim, device=self.device, dtype=torch.float32),
                "M": M,
                "num_heads": num_heads,
                "head_dim": head_dim,
            }
        )

        # Realistic small inference batch (GPT-2-small-style: D=768, 12 heads, dh=64).
        tests.append(self._make_test_case(128, 12, 64))

        # Realistic medium inference batch (LLaMA-2-7B-style: D=4096, 32 heads, dh=128).
        tests.append(self._make_test_case(256, 32, 128))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # LLaMA-2-7B-style attention projection: D=4096, num_heads=32, head_dim=128.
        # M=512 = batch 4 x seq_len 128 (a realistic prefill workload).
        M, num_heads, head_dim = 512, 32, 128
        D = num_heads * head_dim  # 4096
        return {
            "x": RandnTensor((M, D), std=0.1),
            "W_qkv": RandnTensor((3 * D, D), std=0.02),
            "Q": OutTensor((num_heads, M, head_dim)),
            "K": OutTensor((num_heads, M, head_dim)),
            "V": OutTensor((num_heads, M, head_dim)),
            "M": M,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }
