import ctypes
from typing import Any, Dict, List, Optional

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused Residual RMSNorm"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        M: int,
        N: int,
        eps: float,
    ):
        assert x.shape == residual.shape == output.shape == (M, N)
        assert weight.shape == (N,)
        assert x.dtype == residual.dtype == weight.dtype == output.dtype == torch.float32
        assert x.device.type == "cuda"
        assert residual.device.type == "cuda"
        assert weight.device.type == "cuda"
        assert output.device.type == "cuda"

        hidden = x + residual
        inv_rms = torch.rsqrt(torch.mean(hidden * hidden, dim=1, keepdim=True) + eps)
        output.copy_(hidden * inv_rms * weight)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "residual": (ctypes.POINTER(ctypes.c_float), "in"),
            "weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def _make_test_case(
        self,
        M: int,
        N: int,
        *,
        zero_hidden: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        if seed is not None:
            torch.manual_seed(seed)

        if zero_hidden:
            x = torch.empty(M, N, device=device, dtype=dtype).uniform_(-2.0, 2.0)
            residual = -x.clone()
        else:
            x = torch.empty(M, N, device=device, dtype=dtype).uniform_(-3.0, 3.0)
            residual = torch.empty(M, N, device=device, dtype=dtype).uniform_(-3.0, 3.0)

        weight = torch.empty(N, device=device, dtype=dtype).uniform_(0.5, 1.5)
        output = torch.empty(M, N, device=device, dtype=dtype)
        return {
            "x": x,
            "residual": residual,
            "weight": weight,
            "output": output,
            "M": M,
            "N": N,
            "eps": 1e-5,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, N = 2, 4
        return {
            "x": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]],
                device=device,
                dtype=dtype,
            ),
            "residual": torch.tensor(
                [[0.5, -0.5, 1.0, -1.0], [1.0, 2.0, -1.0, 0.0]],
                device=device,
                dtype=dtype,
            ),
            "weight": torch.tensor([1.0, 0.5, 2.0, -1.0], device=device, dtype=dtype),
            "output": torch.empty(M, N, device=device, dtype=dtype),
            "M": M,
            "N": N,
            "eps": 1e-5,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        return [
            self._make_test_case(1, 1, seed=0),
            self._make_test_case(1, 2, seed=1),
            self._make_test_case(2, 4, seed=2),
            self._make_test_case(4, 8, zero_hidden=True, seed=3),
            self._make_test_case(8, 16, seed=4),
            self._make_test_case(16, 64, seed=5),
            self._make_test_case(7, 30, seed=6),
            self._make_test_case(5, 100, seed=7),
            self._make_test_case(32, 256, seed=8),
            self._make_test_case(64, 1024, seed=9),
        ]

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(4096, 1024, seed=42)
