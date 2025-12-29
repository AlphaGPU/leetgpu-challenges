import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Rotary Positional Embedding",
            atol=1e-4,
            rtol=1e-4,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        Q: torch.Tensor,
        Cos: torch.Tensor,
        Sin: torch.Tensor,
        Output: torch.Tensor,
        M: int,
        D: int,
    ):
        assert Q.shape == (M, D)
        assert Cos.shape == (M, D)
        assert Sin.shape == (M, D)
        assert Output.shape == (M, D)

        # rotate_half implementation
        # Split the last dimension into two halves
        x1 = Q[..., : D // 2]
        x2 = Q[..., D // 2 :]
        # Concatenate -x2 and x1
        rotated_Q = torch.cat((-x2, x1), dim=-1)

        # RoPE calculation
        # Output = Q * Cos + rotate_half(Q) * Sin
        result = (Q * Cos) + (rotated_Q * Sin)

        Output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "Cos": (ctypes.POINTER(ctypes.c_float), "in"),
            "Sin": (ctypes.POINTER(ctypes.c_float), "in"),
            "Output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        M = 1024
        D = 64
        dtype = torch.float32

        Q = torch.randn(M, D, device="cuda", dtype=dtype)
        Cos = torch.randn(M, D, device="cuda", dtype=dtype)
        Sin = torch.randn(M, D, device="cuda", dtype=dtype)
        Output = torch.zeros(M, D, device="cuda", dtype=dtype)

        return {
            "Q": Q,
            "Cos": Cos,
            "Sin": Sin,
            "Output": Output,
            "M": M,
            "D": D,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        dtype = torch.float32

        # Test 1: Small input
        M = 4
        D = 4
        tests.append(
            {
                "Q": torch.randn(M, D, device="cuda", dtype=dtype),
                "Cos": torch.randn(M, D, device="cuda", dtype=dtype),
                "Sin": torch.randn(M, D, device="cuda", dtype=dtype),
                "Output": torch.zeros(M, D, device="cuda", dtype=dtype),
                "M": M,
                "D": D,
            }
        )

        # Test 2: Larger input
        M = 128
        D = 64
        tests.append(
            {
                "Q": torch.randn(M, D, device="cuda", dtype=dtype),
                "Cos": torch.randn(M, D, device="cuda", dtype=dtype),
                "Sin": torch.randn(M, D, device="cuda", dtype=dtype),
                "Output": torch.zeros(M, D, device="cuda", dtype=dtype),
                "M": M,
                "D": D,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M = 1024 * 1024  # 1M tokens
        D = 128
        dtype = torch.float32
        return {
            "Q": torch.randn(M, D, device="cuda", dtype=dtype),
            "Cos": torch.randn(M, D, device="cuda", dtype=dtype),
            "Sin": torch.randn(M, D, device="cuda", dtype=dtype),
            "Output": torch.zeros(M, D, device="cuda", dtype=dtype),
            "M": M,
            "D": D,
        }
