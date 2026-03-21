import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="LoRA Linear",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        x: torch.Tensor,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        output: torch.Tensor,
        batch: int,
        d_in: int,
        d_out: int,
        rank: int,
        lora_scale: float,
    ):
        assert x.shape == (batch, d_in)
        assert W.shape == (d_out, d_in)
        assert A.shape == (rank, d_in)
        assert B.shape == (d_out, rank)
        assert output.shape == (batch, d_out)
        assert x.dtype == W.dtype == A.dtype == B.dtype == output.dtype == torch.float32
        assert x.device.type == "cuda"
        assert W.device.type == "cuda"
        assert A.device.type == "cuda"
        assert B.device.type == "cuda"
        assert output.device.type == "cuda"

        # Base linear: output = x @ W^T
        base = torch.mm(x, W.t())

        # LoRA path: delta = lora_scale * (x @ A^T) @ B^T
        lora_hidden = torch.mm(x, A.t())  # (batch, rank)
        delta = torch.mm(lora_hidden, B.t())  # (batch, d_out)

        output.copy_(base + lora_scale * delta)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "W": (ctypes.POINTER(ctypes.c_float), "in"),
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "batch": (ctypes.c_int, "in"),
            "d_in": (ctypes.c_int, "in"),
            "d_out": (ctypes.c_int, "in"),
            "rank": (ctypes.c_int, "in"),
            "lora_scale": (ctypes.c_float, "in"),
        }

    def _make_test_case(self, batch, d_in, d_out, rank, lora_scale=0.5, zero_x=False):
        dtype = torch.float32
        device = "cuda"
        if zero_x:
            x = torch.zeros(batch, d_in, device=device, dtype=dtype)
        else:
            x = torch.randn(batch, d_in, device=device, dtype=dtype)
        W = torch.randn(d_out, d_in, device=device, dtype=dtype) * 0.02
        A = torch.randn(rank, d_in, device=device, dtype=dtype) * 0.02
        B = torch.zeros(d_out, rank, device=device, dtype=dtype)
        output = torch.zeros(batch, d_out, device=device, dtype=dtype)
        return {
            "x": x,
            "W": W,
            "A": A,
            "B": B,
            "output": output,
            "batch": batch,
            "d_in": d_in,
            "d_out": d_out,
            "rank": rank,
            "lora_scale": lora_scale,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"
        x = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.0, 1.0, 1.0, -1.0]], device=device, dtype=dtype)
        W = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        A = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype)
        B = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        output = torch.zeros(2, 3, device=device, dtype=dtype)
        return {
            "x": x,
            "W": W,
            "A": A,
            "B": B,
            "output": output,
            "batch": 2,
            "d_in": 4,
            "d_out": 3,
            "rank": 2,
            "lora_scale": 0.5,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: batch=1, tiny dimensions
        tests.append(self._make_test_case(1, 4, 4, 1))

        # Edge case: zero input
        tests.append(self._make_test_case(2, 8, 8, 2, zero_x=True))

        # Edge case: rank=1 (minimum LoRA rank)
        tests.append(self._make_test_case(4, 16, 16, 1))

        # Power-of-2 dimensions
        tests.append(self._make_test_case(16, 64, 64, 8))

        # Power-of-2, non-square
        tests.append(self._make_test_case(32, 128, 64, 16))

        # Non-power-of-2 dimensions
        tests.append(self._make_test_case(30, 100, 100, 4))

        # Non-power-of-2, mixed
        tests.append(self._make_test_case(7, 255, 128, 8))

        # Realistic small: LLM feed-forward style
        tests.append(self._make_test_case(64, 512, 512, 16, lora_scale=0.125))

        # Negative inputs
        tests.append(
            {
                "x": torch.full((4, 32), -1.0, device="cuda", dtype=torch.float32),
                "W": torch.randn(32, 32, device="cuda", dtype=torch.float32) * 0.02,
                "A": torch.randn(8, 32, device="cuda", dtype=torch.float32) * 0.02,
                "B": torch.randn(32, 8, device="cuda", dtype=torch.float32) * 0.02,
                "output": torch.zeros(4, 32, device="cuda", dtype=torch.float32),
                "batch": 4,
                "d_in": 32,
                "d_out": 32,
                "rank": 8,
                "lora_scale": 1.0,
            }
        )

        # Larger realistic: transformer hidden size
        tests.append(self._make_test_case(128, 1024, 1024, 32, lora_scale=0.0625))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # LLaMA-style: d_in=d_out=4096, rank=64, batch=256
        return self._make_test_case(256, 4096, 4096, 64, lora_scale=0.015625)
