import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "QK-Norm Multi-Head Attention"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        output: torch.Tensor,
        N: int,
        d_model: int,
        h: int,
        eps: float,
    ):
        d_head = d_model // h
        assert Q.shape == (N, d_model)
        assert K.shape == (N, d_model)
        assert V.shape == (N, d_model)
        assert q_weight.shape == (d_head,)
        assert k_weight.shape == (d_head,)
        assert output.shape == (N, d_model)
        assert (
            Q.dtype
            == K.dtype
            == V.dtype
            == q_weight.dtype
            == k_weight.dtype
            == output.dtype
            == torch.float32
        )

        Q_h = Q.view(N, h, d_head)
        K_h = K.view(N, h, d_head)
        V_h = V.view(N, h, d_head)

        q_rms = torch.sqrt(torch.mean(Q_h**2, dim=-1, keepdim=True) + eps)
        k_rms = torch.sqrt(torch.mean(K_h**2, dim=-1, keepdim=True) + eps)
        Q_norm = (Q_h / q_rms) * q_weight
        K_norm = (K_h / k_rms) * k_weight

        Q_t = Q_norm.transpose(0, 1)
        K_t = K_norm.transpose(0, 1)
        V_t = V_h.transpose(0, 1)

        scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) / math.sqrt(d_head)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, V_t)

        output.copy_(attn.transpose(0, 1).contiguous().view(N, d_model))

    def reference_impl_jax(self, Q, K, V, q_weight, k_weight, N, d_model, h, eps):
        import jax
        import jax.numpy as jnp

        d_head = d_model // h

        Q_h = Q.reshape(N, h, d_head)
        K_h = K.reshape(N, h, d_head)
        V_h = V.reshape(N, h, d_head)

        q_rms = jnp.sqrt(jnp.mean(Q_h**2, axis=-1, keepdims=True) + eps)
        k_rms = jnp.sqrt(jnp.mean(K_h**2, axis=-1, keepdims=True) + eps)
        Q_norm = (Q_h / q_rms) * q_weight
        K_norm = (K_h / k_rms) * k_weight

        Q_t = jnp.transpose(Q_norm, (1, 0, 2))
        K_t = jnp.transpose(K_norm, (1, 0, 2))
        V_t = jnp.transpose(V_h, (1, 0, 2))

        scores = jnp.matmul(
            Q_t, jnp.transpose(K_t, (0, 2, 1)), precision=jax.lax.Precision.HIGHEST
        ) / math.sqrt(d_head)
        weights = jax.nn.softmax(scores, axis=-1)
        attn = jnp.matmul(weights, V_t, precision=jax.lax.Precision.HIGHEST)

        return jnp.transpose(attn, (1, 0, 2)).reshape(N, d_model)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "q_weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "k_weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "h": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def _make_test_case(
        self,
        N,
        d_model,
        h,
        eps=1e-6,
        seed=None,
        zero_qkv=False,
        negative=False,
    ):
        device = self.device
        dtype = torch.float32
        if seed is not None:
            torch.manual_seed(seed)
        if zero_qkv:
            Q = torch.zeros(N, d_model, device=device, dtype=dtype)
            K = torch.zeros(N, d_model, device=device, dtype=dtype)
            V = torch.zeros(N, d_model, device=device, dtype=dtype)
        elif negative:
            Q = torch.empty(N, d_model, device=device, dtype=dtype).uniform_(-2.0, -0.1)
            K = torch.empty(N, d_model, device=device, dtype=dtype).uniform_(-2.0, -0.1)
            V = torch.empty(N, d_model, device=device, dtype=dtype).uniform_(-2.0, -0.1)
        else:
            Q = torch.randn(N, d_model, device=device, dtype=dtype)
            K = torch.randn(N, d_model, device=device, dtype=dtype)
            V = torch.randn(N, d_model, device=device, dtype=dtype)
        d_head = d_model // h
        q_weight = torch.empty(d_head, device=device, dtype=dtype).uniform_(0.5, 1.5)
        k_weight = torch.empty(d_head, device=device, dtype=dtype).uniform_(0.5, 1.5)
        output = torch.empty(N, d_model, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "q_weight": q_weight,
            "k_weight": k_weight,
            "output": output,
            "N": N,
            "d_model": d_model,
            "h": h,
            "eps": eps,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        Q = torch.tensor([[1.0, 0.0, 2.0, 3.0], [0.5, -1.0, 0.0, 2.0]], device=device, dtype=dtype)
        K = torch.tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 2.0, -1.0]], device=device, dtype=dtype)
        V = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]], device=device, dtype=dtype)
        q_weight = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
        k_weight = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
        output = torch.empty(2, 4, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "q_weight": q_weight,
            "k_weight": k_weight,
            "output": output,
            "N": 2,
            "d_model": 4,
            "h": 2,
            "eps": 1e-6,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        # Edge: single token, single head
        tests.append(self._make_test_case(1, 4, 1, seed=0))
        # Edge: two tokens, one head
        tests.append(self._make_test_case(2, 8, 1, seed=1))
        # Edge: zero Q/K/V (uniform attention with zero output)
        tests.append(self._make_test_case(4, 16, 2, zero_qkv=True, seed=2))
        # All-negative values
        tests.append(self._make_test_case(4, 16, 4, negative=True, seed=3))
        # Power-of-2 sizes
        tests.append(self._make_test_case(16, 64, 4, seed=4))
        tests.append(self._make_test_case(64, 128, 8, seed=5))
        # Non-power-of-2 sequence lengths
        tests.append(self._make_test_case(30, 64, 4, seed=6))
        tests.append(self._make_test_case(100, 128, 8, seed=7))
        # Realistic sizes
        tests.append(self._make_test_case(256, 512, 8, seed=8))
        tests.append(self._make_test_case(512, 1024, 16, seed=9))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(2048, 2048, 16, seed=42)
