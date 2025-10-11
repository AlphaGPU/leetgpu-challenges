import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Matrix Copy",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, N: int):
        assert A.shape == (N, N)
        assert B.shape == (N, N)
        assert A.dtype == B.dtype
        assert A.device == B.device

        # Copy matrix A to B
        B[:] = A

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype)
        B = torch.empty(2, 2, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "N": 2
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32

        test_cases = []

        # basic_2x2
        test_cases.append({
            "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype),
            "B": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2
        })

        # all_zeros_4x4
        test_cases.append({
            "A": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "B": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "N": 4
        })

        # identity_3x3
        test_cases.append({
            "A": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cuda", dtype=dtype),
            "B": torch.zeros((3, 3), device="cuda", dtype=dtype),
            "N": 3
        })

        # negative_values_2x2
        test_cases.append({
            "A": torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device="cuda", dtype=dtype),
            "B": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2
        })

        # large_N_16x16
        test_cases.append({
            "A": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.zeros((16, 16), device="cuda", dtype=dtype),
            "N": 16
        })

        # single_element
        test_cases.append({
            "A": torch.tensor([[42.0]], device="cuda", dtype=dtype),
            "B": torch.zeros((1, 1), device="cuda", dtype=dtype),
            "N": 1
        })

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4096
        return {
            "A": torch.empty(N, N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "B": torch.empty(N, N, device="cuda", dtype=dtype),
            "N": N
        }
