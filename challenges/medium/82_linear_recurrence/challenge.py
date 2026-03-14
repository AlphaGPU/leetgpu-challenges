import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Linear Recurrence",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor,
        B: int,
        L: int,
    ):
        assert a.shape == (B, L)
        assert x.shape == (B, L)
        assert h.shape == (B, L)
        assert a.dtype == x.dtype == h.dtype == torch.float32
        assert a.device.type == "cuda"
        assert x.device.type == "cuda"
        assert h.device.type == "cuda"

        out = torch.empty_like(x)
        out[:, 0] = x[:, 0]
        for t in range(1, L):
            out[:, t] = a[:, t] * out[:, t - 1] + x[:, t]
        h.copy_(out)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "a": (ctypes.POINTER(ctypes.c_float), "in"),
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "h": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "L": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, B, L, zero_inputs=False, zero_a=False, unit_a=False):
        device = "cuda"
        dtype = torch.float32
        if zero_inputs:
            a = torch.zeros(B, L, device=device, dtype=dtype)
            x = torch.zeros(B, L, device=device, dtype=dtype)
        elif zero_a:
            a = torch.zeros(B, L, device=device, dtype=dtype)
            x = torch.randn(B, L, device=device, dtype=dtype)
        elif unit_a:
            a = torch.ones(B, L, device=device, dtype=dtype)
            x = torch.randn(B, L, device=device, dtype=dtype)
        else:
            a = torch.rand(B, L, device=device, dtype=dtype)
            x = torch.randn(B, L, device=device, dtype=dtype)
        h = torch.empty(B, L, device=device, dtype=dtype)
        return {"a": a, "x": x, "h": h, "B": B, "L": L}

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        dtype = torch.float32
        a = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        x = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        h = torch.empty(2, 4, device=device, dtype=dtype)
        return {"a": a, "x": x, "h": h, "B": 2, "L": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: single element
        tests.append(self._make_test_case(1, 1))

        # Edge case: two elements
        tests.append(self._make_test_case(1, 2))

        # Zero inputs
        tests.append(self._make_test_case(4, 4, zero_inputs=True))

        # a=0 everywhere: h[t] = x[t] (no recurrence)
        tests.append(self._make_test_case(4, 16, zero_a=True))

        # a=1 everywhere: h[t] = prefix sum of x
        tests.append(self._make_test_case(4, 16, unit_a=True))

        # Power-of-2 sequence length
        tests.append(self._make_test_case(8, 32))

        # Power-of-2 sequence length, larger
        tests.append(self._make_test_case(8, 256))

        # Non-power-of-2 sequence length
        tests.append(self._make_test_case(4, 30))

        # Non-power-of-2 sequence length, larger
        tests.append(self._make_test_case(8, 100))

        # Realistic size (SSM hidden state)
        tests.append(self._make_test_case(16, 1024))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # B=64 sequences, L=16384 tokens — typical long-context SSM workload
        return self._make_test_case(64, 16384)
