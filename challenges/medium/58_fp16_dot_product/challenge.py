import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="FP16 Dot Product",
            atol=5e-2,
            rtol=5e-2,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
        assert A.shape == (N,)
        assert B.shape == (N,)
        assert result.shape == (1,)
        # Use FP32 for accumulation, then convert to FP16
        A_f32 = A.to(torch.float32)
        B_f32 = B.to(torch.float32)
        result_f32 = torch.dot(A_f32, B_f32)
        result[0] = result_f32.to(torch.float16)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "B": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "result": (ctypes.POINTER(ctypes.c_uint16), "out"),
            "N": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float16
        A = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        B = torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype)
        result = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "result": result,
            "N": 4
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float16
        tests = []
        # basic_small
        tests.append({
            "A": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype),
            "B": torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "result": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4
        })
        # all_zeros
        tests.append({
            "A": torch.tensor([0.0] * 16, device="cuda", dtype=dtype),
            "B": torch.tensor([0.0] * 16, device="cuda", dtype=dtype),
            "result": torch.empty(1, device="cuda", dtype=dtype),
            "N": 16
        })
        # negative_numbers
        tests.append({
            "A": torch.tensor([-1.0, -2.0, -3.0, -4.0], device="cuda", dtype=dtype),
            "B": torch.tensor([-5.0, -6.0, -7.0, -8.0], device="cuda", dtype=dtype),
            "result": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4
        })
        # mixed_positive_negative
        tests.append({
            "A": torch.tensor([1.0, -2.0, 3.0, -4.0], device="cuda", dtype=dtype),
            "B": torch.tensor([-1.0, 2.0, -3.0, 4.0], device="cuda", dtype=dtype),
            "result": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4
        })
        # orthogonal_vectors
        tests.append({
            "A": torch.tensor([1.0, 0.0, 0.0], device="cuda", dtype=dtype),
            "B": torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=dtype),
            "result": torch.empty(1, device="cuda", dtype=dtype),
            "N": 3
        })
        # medium_sized_vector
        tests.append({
            "A": torch.empty(1000, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "B": torch.empty(1000, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "result": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1000
        })
        # large_vector
        tests.append({
            "A": torch.empty(10000, device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "B": torch.empty(10000, device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "result": torch.empty(1, device="cuda", dtype=dtype),
            "N": 10000
        })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float16
        N = 100000000
        A = torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        B = torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        result = torch.zeros(1, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "result": result,
            "N": N
        }
