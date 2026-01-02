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
        cos: torch.Tensor,
        sin: torch.Tensor,
        output: torch.Tensor,
        M: int,
        D: int,
    ):
        assert Q.shape == (M, D)
        assert cos.shape == (M, D)
        assert sin.shape == (M, D)
        assert output.shape == (M, D)

        # rotate_half implementation
        # Split the last dimension into two halves
        x1 = Q[..., : D // 2]
        x2 = Q[..., D // 2 :]
        # Concatenate -x2 and x1
        rotated_Q = torch.cat((-x2, x1), dim=-1)

        # RoPE calculation
        # Output = Q * Cos + rotate_half(Q) * Sin
        result = (Q * cos) + (rotated_Q * sin)

        output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "cos": (ctypes.POINTER(ctypes.c_float), "in"),
            "sin": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
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
            "cos": Cos,
            "sin": Sin,
            "output": Output,
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
                "cos": torch.randn(M, D, device="cuda", dtype=dtype),
                "sin": torch.randn(M, D, device="cuda", dtype=dtype),
                "output": torch.zeros(M, D, device="cuda", dtype=dtype),
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
                "cos": torch.randn(M, D, device="cuda", dtype=dtype),
                "sin": torch.randn(M, D, device="cuda", dtype=dtype),
                "output": torch.zeros(M, D, device="cuda", dtype=dtype),
                "M": M,
                "D": D,
            }
        )

        # zero_matrices: outputs should remain zero when inputs are zero
        tests.append(
            {
                "Q": torch.zeros((3, 6), device="cuda", dtype=dtype),
                "cos": torch.zeros((3, 6), device="cuda", dtype=dtype),
                "sin": torch.zeros((3, 6), device="cuda", dtype=dtype),
                "output": torch.zeros(3, 6, device="cuda", dtype=dtype),
                "M": 3,
                "D": 6,
            }
        )

        # minimal_dims: smallest even D that still allows rotation
        tests.append(
            {
                "Q": torch.randn((1, 2), device="cuda", dtype=dtype),
                "cos": torch.randn((1, 2), device="cuda", dtype=dtype),
                "sin": torch.randn((1, 2), device="cuda", dtype=dtype),
                "output": torch.zeros(1, 2, device="cuda", dtype=dtype),
                "M": 1,
                "D": 2,
            }
        )

        # mixed_values: negative and positive entries
        tests.append(
            {
                "Q": torch.tensor(
                    [[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]],
                    device="cuda",
                    dtype=dtype,
                ),
                "cos": torch.tensor(
                    [[0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.3, 0.4]],
                    device="cuda",
                    dtype=dtype,
                ),
                "sin": torch.tensor(
                    [[0.5, -0.5, 0.5, -0.5], [0.4, -0.3, 0.2, -0.1]],
                    device="cuda",
                    dtype=dtype,
                ),
                "output": torch.zeros(2, 4, device="cuda", dtype=dtype),
                "M": 2,
                "D": 4,
            }
        )

        # large_matrices: random uniform values for stress testing
        tests.append(
            {
                "Q": torch.empty((256, 128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
                "cos": torch.empty((256, 128), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "sin": torch.empty((256, 128), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros(256, 128, device="cuda", dtype=dtype),
                "M": 256,
                "D": 128,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M = 1024 * 1024  # 1M tokens
        D = 128
        dtype = torch.float32
        return {
            "Q": torch.randn(M, D, device="cuda", dtype=dtype),
            "cos": torch.randn(M, D, device="cuda", dtype=dtype),
            "sin": torch.randn(M, D, device="cuda", dtype=dtype),
            "output": torch.zeros(M, D, device="cuda", dtype=dtype),
            "M": M,
            "D": D,
        }
