import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Grouped Query Attention",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
    ):
        assert Q.shape == (num_q_heads, seq_len, head_dim)
        assert K.shape == (num_kv_heads, seq_len, head_dim)
        assert V.shape == (num_kv_heads, seq_len, head_dim)
        assert output.shape == (num_q_heads, seq_len, head_dim)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32
        assert Q.device.type == "cuda"
        assert K.device.type == "cuda"
        assert V.device.type == "cuda"
        assert output.device.type == "cuda"
        assert num_q_heads % num_kv_heads == 0

        num_groups = num_q_heads // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        # Expand K, V from (num_kv_heads, seq_len, head_dim)
        # to (num_q_heads, seq_len, head_dim) by repeating each KV head num_groups times
        K_expanded = K.repeat_interleave(num_groups, dim=0)
        V_expanded = V.repeat_interleave(num_groups, dim=0)

        # Scaled dot-product attention: (num_q_heads, seq_len, seq_len)
        scores = torch.bmm(Q, K_expanded.transpose(1, 2)) * scale

        # Softmax over the key dimension
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values: (num_q_heads, seq_len, head_dim)
        output.copy_(torch.bmm(attn_weights, V_expanded))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_q_heads": (ctypes.c_int, "in"),
            "num_kv_heads": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, num_q_heads, num_kv_heads, seq_len, head_dim, zero_inputs=False):
        dtype = torch.float32
        device = "cuda"
        if zero_inputs:
            Q = torch.zeros(num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
            K = torch.zeros(num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
            V = torch.zeros(num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        else:
            Q = torch.randn(num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
            K = torch.randn(num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
            V = torch.randn(num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        output = torch.zeros(num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        dtype = torch.float32
        device = "cuda"
        num_q_heads = 4
        num_kv_heads = 2
        seq_len = 3
        head_dim = 4

        Q = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
                [[-1.0, 0.0, 0.5, 0.0], [0.0, -1.0, 0.0, 0.5], [0.5, 0.0, -1.0, 0.0]],
                [[0.0, 0.5, 0.0, -1.0], [0.5, 0.0, 0.0, -1.0], [0.0, 0.0, 0.5, 0.5]],
            ],
            device=device,
            dtype=dtype,
        )
        K = torch.tensor(
            [
                [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                [[0.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]],
            ],
            device=device,
            dtype=dtype,
        )
        V = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                [[-1.0, -2.0, -3.0, -4.0], [2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]],
            ],
            device=device,
            dtype=dtype,
        )
        output = torch.zeros(num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: MQA (num_kv_heads=1), single token
        tests.append(self._make_test_case(4, 1, 1, 8))

        # Edge case: GQA with groups=2, tiny seq
        tests.append(self._make_test_case(2, 1, 2, 4))

        # Zero inputs
        tests.append(self._make_test_case(4, 2, 4, 8, zero_inputs=True))

        # Power-of-2: groups=4 (LLaMA-3 style ratio)
        tests.append(self._make_test_case(8, 2, 16, 32))

        # Power-of-2: seq_len=32, head_dim=64
        tests.append(self._make_test_case(4, 2, 32, 64))

        # Non-power-of-2 seq_len
        tests.append(self._make_test_case(4, 2, 30, 32))

        # Non-power-of-2 seq_len, different grouping
        tests.append(self._make_test_case(6, 3, 100, 32))

        # GQA groups=8 (Mistral style), seq_len=255
        tests.append(self._make_test_case(8, 1, 255, 64))

        # MHA equivalent (num_q_heads == num_kv_heads)
        tests.append(self._make_test_case(8, 8, 64, 32))

        # Realistic small inference batch
        tests.append(self._make_test_case(8, 2, 128, 64))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # LLaMA-3 8B style: 32 Q heads, 8 KV heads, head_dim=128
        return self._make_test_case(32, 8, 1024, 128)
