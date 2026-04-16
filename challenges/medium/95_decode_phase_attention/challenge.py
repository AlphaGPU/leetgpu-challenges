import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Decode-Phase Attention",
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
        batch_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        cache_len: int,
        head_dim: int,
    ):
        assert Q.shape == (batch_size, num_q_heads, head_dim)
        assert K.shape == (batch_size, num_kv_heads, cache_len, head_dim)
        assert V.shape == (batch_size, num_kv_heads, cache_len, head_dim)
        assert output.shape == (batch_size, num_q_heads, head_dim)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32
        assert Q.device.type == "cuda"
        assert K.device.type == "cuda"
        assert V.device.type == "cuda"
        assert output.device.type == "cuda"
        assert num_q_heads % num_kv_heads == 0

        scale = 1.0 / math.sqrt(head_dim)
        num_groups = num_q_heads // num_kv_heads

        # Expand K and V from (B, Hkv, T, D) to (B, Hq, T, D)
        K_exp = K.repeat_interleave(num_groups, dim=1)
        V_exp = V.repeat_interleave(num_groups, dim=1)

        # scores: (B, Hq, T) = Q(B, Hq, 1, D) @ K^T(B, Hq, D, T) -> squeeze
        Q_unsq = Q.unsqueeze(2)  # (B, Hq, 1, D)
        scores = torch.matmul(Q_unsq, K_exp.transpose(2, 3)).squeeze(2) * scale

        # Softmax over the cache dimension
        weights = torch.softmax(scores, dim=-1)  # (B, Hq, T)

        # Weighted sum of values: (B, Hq, T) x (B, Hq, T, D) -> (B, Hq, D)
        out = torch.matmul(weights.unsqueeze(2), V_exp).squeeze(2)
        output.copy_(out)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "batch_size": (ctypes.c_int, "in"),
            "num_q_heads": (ctypes.c_int, "in"),
            "num_kv_heads": (ctypes.c_int, "in"),
            "cache_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        batch_size,
        num_q_heads,
        num_kv_heads,
        cache_len,
        head_dim,
        zero_inputs=False,
    ):
        dtype = torch.float32
        device = "cuda"
        if zero_inputs:
            Q = torch.zeros(batch_size, num_q_heads, head_dim, device=device, dtype=dtype)
            K = torch.zeros(
                batch_size, num_kv_heads, cache_len, head_dim, device=device, dtype=dtype
            )
            V = torch.zeros(
                batch_size, num_kv_heads, cache_len, head_dim, device=device, dtype=dtype
            )
        else:
            Q = torch.randn(batch_size, num_q_heads, head_dim, device=device, dtype=dtype)
            K = torch.randn(
                batch_size, num_kv_heads, cache_len, head_dim, device=device, dtype=dtype
            )
            V = torch.randn(
                batch_size, num_kv_heads, cache_len, head_dim, device=device, dtype=dtype
            )
        output = torch.zeros(batch_size, num_q_heads, head_dim, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "batch_size": batch_size,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "cache_len": cache_len,
            "head_dim": head_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"
        Q = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        K = torch.tensor(
            [[[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]]]],
            device=device,
            dtype=dtype,
        )
        V = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]],
            device=device,
            dtype=dtype,
        )
        output = torch.zeros(1, 2, 4, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "batch_size": 1,
            "num_q_heads": 2,
            "num_kv_heads": 1,
            "cache_len": 3,
            "head_dim": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: single head, single cache position
        tests.append(self._make_test_case(1, 1, 1, 1, 8))

        # Edge case: zero inputs (softmax of uniform zeros → uniform weights)
        tests.append(self._make_test_case(1, 2, 1, 4, 8, zero_inputs=True))

        # MQA (num_kv_heads=1): all query heads share one KV head
        tests.append(self._make_test_case(2, 4, 1, 16, 16))

        # GQA with groups=2, short cache
        tests.append(self._make_test_case(2, 4, 2, 2, 8))

        # MHA equivalent (num_kv_heads == num_q_heads)
        tests.append(self._make_test_case(1, 4, 4, 16, 32))

        # Power-of-2 cache length
        tests.append(self._make_test_case(2, 8, 2, 64, 32))

        # Power-of-2 larger cache
        tests.append(self._make_test_case(2, 8, 2, 256, 64))

        # Non-power-of-2 cache length
        tests.append(self._make_test_case(2, 4, 2, 30, 32))

        # Non-power-of-2, larger
        tests.append(self._make_test_case(4, 4, 2, 100, 32))

        # Realistic small inference: LLaMA-3 8B style heads
        tests.append(self._make_test_case(2, 32, 8, 1024, 128))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # LLaMA-3 8B: 32 Q heads, 8 KV heads, head_dim=128, long context
        return self._make_test_case(4, 32, 8, 16384, 128)
