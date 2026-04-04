import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Fused Residual Add and RMS Norm",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        N: int,
        C: int,
        eps: float,
    ):
        assert x.shape == (N, C)
        assert residual.shape == (N, C)
        assert weight.shape == (C,)
        assert out.shape == (N, C)
        assert x.dtype == residual.dtype == weight.dtype == out.dtype == torch.float32
        assert x.device.type == "cuda"
        assert residual.device.type == "cuda"
        assert weight.device.type == "cuda"
        assert out.device.type == "cuda"

        z = x + residual
        rms = torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True) + eps)
        out.copy_(z / rms * weight)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "residual": (ctypes.POINTER(ctypes.c_float), "in"),
            "weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "out": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def _make_test_case(self, N, C, eps=1e-5, zero_x=False, zero_residual=False, negative=False):
        device = "cuda"
        dtype = torch.float32
        if zero_x:
            x = torch.zeros(N, C, device=device, dtype=dtype)
        elif negative:
            x = torch.empty(N, C, device=device, dtype=dtype).uniform_(-2.0, -0.1)
        else:
            x = torch.randn(N, C, device=device, dtype=dtype)
        if zero_residual:
            residual = torch.zeros(N, C, device=device, dtype=dtype)
        else:
            residual = torch.randn(N, C, device=device, dtype=dtype)
        weight = torch.empty(C, device=device, dtype=dtype).uniform_(0.5, 1.5)
        out = torch.empty(N, C, device=device, dtype=dtype)
        return {
            "x": x,
            "residual": residual,
            "weight": weight,
            "out": out,
            "N": N,
            "C": C,
            "eps": eps,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        dtype = torch.float32
        x = torch.tensor([[1.0, 0.0, -1.0, 2.0]], device=device, dtype=dtype)
        residual = torch.tensor([[0.5, 1.5, 0.5, -0.5]], device=device, dtype=dtype)
        weight = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device, dtype=dtype)
        out = torch.empty(1, 4, device=device, dtype=dtype)
        return {
            "x": x,
            "residual": residual,
            "weight": weight,
            "out": out,
            "N": 1,
            "C": 4,
            "eps": 1e-5,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: single token, single feature
        tests.append(self._make_test_case(1, 1))

        # Edge case: single token, 4 features
        tests.append(self._make_test_case(1, 4))

        # Edge case: zero x (residual pass-through with normalization)
        tests.append(self._make_test_case(2, 4, zero_x=True))

        # Edge case: zero residual (equivalent to plain RMS norm of x)
        tests.append(self._make_test_case(4, 4, zero_residual=True))

        # Negative values in x
        tests.append(self._make_test_case(4, 8, negative=True))

        # Power-of-2: typical small transformer hidden size
        tests.append(self._make_test_case(16, 64))

        # Power-of-2: medium transformer hidden size
        tests.append(self._make_test_case(32, 256))

        # Non-power-of-2
        tests.append(self._make_test_case(30, 100))

        # Non-power-of-2, larger
        tests.append(self._make_test_case(100, 255))

        # Realistic: batch of tokens, LLM-style hidden size
        tests.append(self._make_test_case(128, 512))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # 4096 tokens (e.g. batch=4, seq_len=1024), C=4096 (LLaMA-7B hidden size)
        return self._make_test_case(4096, 4096)
