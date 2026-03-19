import ctypes
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="SwiGLU MLP Block",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        x: torch.Tensor,
        W_gate: torch.Tensor,
        W_up: torch.Tensor,
        W_down: torch.Tensor,
        output: torch.Tensor,
        M: int,
        d_model: int,
        d_ffn: int,
    ):
        assert x.shape == (M, d_model)
        assert W_gate.shape == (d_model, d_ffn)
        assert W_up.shape == (d_model, d_ffn)
        assert W_down.shape == (d_ffn, d_model)
        assert output.shape == (M, d_model)
        assert (
            x.dtype == W_gate.dtype == W_up.dtype == W_down.dtype == output.dtype == torch.float32
        )
        assert x.device.type == "cuda"
        assert W_gate.device.type == "cuda"
        assert W_up.device.type == "cuda"
        assert W_down.device.type == "cuda"
        assert output.device.type == "cuda"

        gate = x @ W_gate  # [M, d_ffn]
        up = x @ W_up  # [M, d_ffn]
        hidden = F.silu(gate) * up  # [M, d_ffn]
        output.copy_(hidden @ W_down)  # [M, d_model]

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_gate": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_up": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_down": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "d_ffn": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, M, d_model, d_ffn, zero_x=False):
        device = "cuda"
        dtype = torch.float32
        if zero_x:
            x = torch.zeros(M, d_model, device=device, dtype=dtype)
        else:
            x = torch.randn(M, d_model, device=device, dtype=dtype) * 0.1
        W_gate = torch.randn(d_model, d_ffn, device=device, dtype=dtype) * 0.02
        W_up = torch.randn(d_model, d_ffn, device=device, dtype=dtype) * 0.02
        W_down = torch.randn(d_ffn, d_model, device=device, dtype=dtype) * 0.02
        output = torch.empty(M, d_model, device=device, dtype=dtype)
        return {
            "x": x,
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "output": output,
            "M": M,
            "d_model": d_model,
            "d_ffn": d_ffn,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        dtype = torch.float32
        M, d_model, d_ffn = 2, 2, 4
        # x: each row is a basis vector
        x = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        # W_gate: [d_model=2, d_ffn=4] — first two columns are identity, rest zeros
        W_gate = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        # W_up: same layout as W_gate
        W_up = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        # W_down: [d_ffn=4, d_model=2] — top 2x2 is identity, rest zeros
        W_down = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        output = torch.empty(M, d_model, device=device, dtype=dtype)
        return {
            "x": x,
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "output": output,
            "M": M,
            "d_model": d_model,
            "d_ffn": d_ffn,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge cases: single row
        tests.append(self._make_test_case(1, 4, 8))

        # Edge case: two rows
        tests.append(self._make_test_case(2, 4, 8))

        # Zero input
        tests.append(self._make_test_case(4, 8, 16, zero_x=True))

        # Power-of-2 sizes
        tests.append(self._make_test_case(16, 32, 64))

        # Power-of-2 larger
        tests.append(self._make_test_case(64, 64, 128))

        # Non-power-of-2 M
        tests.append(self._make_test_case(30, 32, 64))

        # Non-power-of-2 all dims
        tests.append(self._make_test_case(100, 60, 120))

        # Non-power-of-2 M, medium size
        tests.append(self._make_test_case(255, 64, 128))

        # Realistic small inference batch (LLaMA-style ratios)
        tests.append(self._make_test_case(128, 256, 512))

        # Realistic medium inference batch
        tests.append(self._make_test_case(256, 512, 1024))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # LLaMA-3 8B style: d_model=4096, d_ffn=14336, M=512 (batch=4 x seq=128)
        return self._make_test_case(512, 4096, 14336)
