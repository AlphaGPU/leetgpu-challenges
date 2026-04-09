import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Decaying Causal Attention",
            atol=1e-03,
            rtol=1e-03,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        seq_len: int,
        d_model: int,
        gamma: float,
    ):
        assert Q.shape == (seq_len, d_model)
        assert K.shape == (seq_len, d_model)
        assert V.shape == (seq_len, d_model)
        assert output.shape == (seq_len, d_model)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32
        assert Q.device.type == "cuda"
        assert K.device.type == "cuda"
        assert V.device.type == "cuda"
        assert output.device.type == "cuda"

        scale = math.sqrt(d_model)
        positions = torch.arange(seq_len, device=Q.device, dtype=Q.dtype)
        # distances[n, m] = n - m; negative means m is in the future relative to n
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        # causal: zero out future positions; clamp avoids overflow in gamma**negative
        causal = (distances >= 0).to(Q.dtype)
        decay_mask = torch.pow(gamma, distances.clamp(min=0)) * causal
        attn = torch.matmul(Q, K.T) / scale
        output.copy_(torch.matmul(attn * decay_mask, V))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "seq_len": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "gamma": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"
        # Orthogonal K rows → QK^T / sqrt(4) = [[0.5, 0.5], [0.5, 0.5]].
        # With gamma=0.5 decay mask [[1, 0], [0.5, 1]], weighted attn = [[0.5, 0], [0.25, 0.5]].
        # Output row 0 = 0.5 * V[0]; row 1 = 0.25 * V[0] + 0.5 * V[1] = [3, 6, 9, 12].
        Q = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype)
        K = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype)
        V = torch.tensor(
            [[4.0, 8.0, 12.0, 16.0], [4.0, 8.0, 12.0, 16.0]], device=device, dtype=dtype
        )
        output = torch.zeros(2, 4, device=device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "seq_len": 2, "d_model": 4, "gamma": 0.5}

    def _make_test_case(
        self,
        seq_len: int,
        d_model: int,
        gamma: float = 0.9,
        zero_qk: bool = False,
        negative: bool = False,
    ) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"
        if zero_qk:
            Q = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
            K = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
            V = torch.randn(seq_len, d_model, device=device, dtype=dtype)
        elif negative:
            Q = torch.randn(seq_len, d_model, device=device, dtype=dtype).neg()
            K = torch.randn(seq_len, d_model, device=device, dtype=dtype).neg()
            V = torch.randn(seq_len, d_model, device=device, dtype=dtype).neg()
        else:
            Q = torch.randn(seq_len, d_model, device=device, dtype=dtype)
            K = torch.randn(seq_len, d_model, device=device, dtype=dtype)
            V = torch.randn(seq_len, d_model, device=device, dtype=dtype)
        output = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "seq_len": seq_len,
            "d_model": d_model,
            "gamma": gamma,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge: single token (only self-attention possible)
        tests.append(self._make_test_case(1, 4, gamma=0.9))

        # Edge: two tokens (matches example structure)
        tests.append(self._make_test_case(2, 4, gamma=0.5))

        # Edge: gamma=1.0 — no decay, equal weight to all past positions
        tests.append(self._make_test_case(4, 8, gamma=1.0))

        # Edge: small gamma — very sharp recency bias
        tests.append(self._make_test_case(4, 8, gamma=0.1))

        # Zero Q and K: all attention scores are zero → output must be all zeros
        tests.append(self._make_test_case(8, 16, gamma=0.9, zero_qk=True))

        # All-negative Q, K, V
        tests.append(self._make_test_case(16, 16, gamma=0.8, negative=True))

        # Power-of-2 sequence length
        tests.append(self._make_test_case(32, 32, gamma=0.9))

        # Power-of-2, larger
        tests.append(self._make_test_case(64, 64, gamma=0.8))

        # Non-power-of-2 sequence length
        tests.append(self._make_test_case(30, 32, gamma=0.95))

        # Non-power-of-2, larger realistic size
        tests.append(self._make_test_case(100, 64, gamma=0.9))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # Typical LLM head: seq_len=4096, head_dim=64
        return self._make_test_case(4096, 64, gamma=0.9)
