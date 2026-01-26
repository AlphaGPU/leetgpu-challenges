import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Interleave Arrays", atol=1e-05, rtol=1e-05, num_gpus=1, access_tier="free"
        )

    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
        assert A.shape == (N,)
        assert B.shape == (N,)
        assert output.shape == (2 * N,)
        assert A.dtype == B.dtype == output.dtype == torch.float32

        # Interleave: [A[0], B[0], A[1], B[1], ...]
        output[0::2] = A
        output[1::2] = B

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype)
        B = torch.tensor([4.0, 5.0, 6.0], device="cuda", dtype=dtype)
        output = torch.empty(6, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "output": output,
            "N": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Basic example
        tests.append(
            {
                "A": torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype),
                "B": torch.tensor([4.0, 5.0, 6.0], device="cuda", dtype=dtype),
                "output": torch.empty(6, device="cuda", dtype=dtype),
                "N": 3,
            }
        )

        # Single element
        tests.append(
            {
                "A": torch.tensor([1.0], device="cuda", dtype=dtype),
                "B": torch.tensor([2.0], device="cuda", dtype=dtype),
                "output": torch.empty(2, device="cuda", dtype=dtype),
                "N": 1,
            }
        )

        # Two elements
        tests.append(
            {
                "A": torch.tensor([10.0, 20.0], device="cuda", dtype=dtype),
                "B": torch.tensor([30.0, 40.0], device="cuda", dtype=dtype),
                "output": torch.empty(4, device="cuda", dtype=dtype),
                "N": 2,
            }
        )

        # Negative values
        tests.append(
            {
                "A": torch.tensor([-1.0, -2.0, -3.0], device="cuda", dtype=dtype),
                "B": torch.tensor([-4.0, -5.0, -6.0], device="cuda", dtype=dtype),
                "output": torch.empty(6, device="cuda", dtype=dtype),
                "N": 3,
            }
        )

        # Mixed positive and negative
        tests.append(
            {
                "A": torch.tensor([1.0, -2.0, 3.0, -4.0], device="cuda", dtype=dtype),
                "B": torch.tensor([-1.0, 2.0, -3.0, 4.0], device="cuda", dtype=dtype),
                "output": torch.empty(8, device="cuda", dtype=dtype),
                "N": 4,
            }
        )

        # Zeros
        tests.append(
            {
                "A": torch.zeros(5, device="cuda", dtype=dtype),
                "B": torch.ones(5, device="cuda", dtype=dtype),
                "output": torch.empty(10, device="cuda", dtype=dtype),
                "N": 5,
            }
        )

        # Large values
        tests.append(
            {
                "A": torch.tensor([1e10, 1e-10], device="cuda", dtype=dtype),
                "B": torch.tensor([1e-10, 1e10], device="cuda", dtype=dtype),
                "output": torch.empty(4, device="cuda", dtype=dtype),
                "N": 2,
            }
        )

        # Medium size random
        N = 1024
        tests.append(
            {
                "A": torch.randn(N, device="cuda", dtype=dtype),
                "B": torch.randn(N, device="cuda", dtype=dtype),
                "output": torch.empty(2 * N, device="cuda", dtype=dtype),
                "N": N,
            }
        )

        # Larger random
        N = 10000
        tests.append(
            {
                "A": torch.randn(N, device="cuda", dtype=dtype),
                "B": torch.randn(N, device="cuda", dtype=dtype),
                "output": torch.empty(2 * N, device="cuda", dtype=dtype),
                "N": N,
            }
        )

        # Even larger
        N = 100000
        tests.append(
            {
                "A": torch.randn(N, device="cuda", dtype=dtype),
                "B": torch.randn(N, device="cuda", dtype=dtype),
                "output": torch.empty(2 * N, device="cuda", dtype=dtype),
                "N": N,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 25000000  # 25 million elements each, 50 million output
        return {
            "A": torch.randn(N, device="cuda", dtype=dtype),
            "B": torch.randn(N, device="cuda", dtype=dtype),
            "output": torch.empty(2 * N, device="cuda", dtype=dtype),
            "N": N,
        }
