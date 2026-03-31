import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Flash Attention Forward",
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
        num_heads: int,
        seq_len: int,
        head_dim: int,
    ):
        assert Q.shape == (num_heads, seq_len, head_dim)
        assert K.shape == (num_heads, seq_len, head_dim)
        assert V.shape == (num_heads, seq_len, head_dim)
        assert output.shape == (num_heads, seq_len, head_dim)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32
        assert Q.device.type == "cuda"
        assert K.device.type == "cuda"
        assert V.device.type == "cuda"
        assert output.device.type == "cuda"

        scale = 1.0 / math.sqrt(head_dim)
        # scores: (num_heads, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output.copy_(torch.bmm(attn_weights, V))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_heads": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, num_heads, seq_len, head_dim, zero_inputs=False):
        device = "cuda"
        dtype = torch.float32
        if zero_inputs:
            Q = torch.zeros(num_heads, seq_len, head_dim, device=device, dtype=dtype)
            K = torch.zeros(num_heads, seq_len, head_dim, device=device, dtype=dtype)
            V = torch.zeros(num_heads, seq_len, head_dim, device=device, dtype=dtype)
        else:
            Q = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=dtype)
            K = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=dtype)
            V = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=dtype)
        output = torch.empty(num_heads, seq_len, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        dtype = torch.float32
        Q = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        K = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        V = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]],
            device=device,
            dtype=dtype,
        )
        output = torch.empty(1, 3, 4, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_heads": 1,
            "seq_len": 3,
            "head_dim": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge cases: tiny sequences
        tests.append(self._make_test_case(1, 1, 8))
        tests.append(self._make_test_case(2, 2, 8, zero_inputs=True))

        # Edge cases: small sequences, multiple heads
        tests.append(self._make_test_case(4, 3, 16))

        # Power-of-2 sizes
        tests.append(self._make_test_case(1, 16, 32))
        tests.append(self._make_test_case(4, 64, 32))
        tests.append(self._make_test_case(8, 128, 64))

        # Non-power-of-2 sequences
        tests.append(self._make_test_case(2, 30, 32))
        tests.append(self._make_test_case(4, 100, 64))
        tests.append(self._make_test_case(2, 255, 32))

        # Realistic size
        tests.append(self._make_test_case(8, 512, 64))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(16, 4096, 64)
