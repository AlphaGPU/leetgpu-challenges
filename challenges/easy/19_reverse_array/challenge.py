import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Reverse Array",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, N: int):
        assert input.shape == (N,)
        assert input.dtype == torch.float32

        # Reverse the array in-place
        input[:] = torch.flip(input, [0])

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "inout"),
            "N": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        return {
            "input": input_tensor,
            "N": 4
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32

        test_cases = []

        # Fixed value test cases
        test_cases.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype),
            "N": 4
        })

        test_cases.append({
            "input": torch.tensor([42.0], device="cuda", dtype=dtype),
            "N": 1
        })

        test_cases.append({
            "input": torch.tensor(
                [0.0] * 16,
                device="cuda",
                dtype=dtype
            ),
            "N": 16
        })

        test_cases.append({
            "input": torch.tensor(
                [
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                    20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0
                ],
                device="cuda",
                dtype=dtype
            ),
            "N": 30
        })

        test_cases.append({
            "input": torch.tensor([-1.0, -2.0, -3.0, -4.0], device="cuda", dtype=dtype),
            "N": 4
        })

        test_cases.append({
            "input": torch.tensor([1.0, -2.0, 3.0, -4.0], device="cuda", dtype=dtype),
            "N": 4
        })

        test_cases.append({
            "input": torch.tensor(
                [0.000001, 0.0000001, 0.00000001, 0.000000001],
                device="cuda",
                dtype=dtype
            ),
            "N": 4
        })

        test_cases.append({
            "input": torch.tensor(
                [1000000.0, 10000000.0, -1000000.0, -10000000.0],
                device="cuda",
                dtype=dtype
            ),
            "N": 4
        })

        # Random range test cases
        test_cases.append({
            "input": torch.empty(32, device="cuda", dtype=dtype).uniform_(0.0, 32.0),
            "N": 32
        })

        test_cases.append({
            "input": torch.empty(1000, device="cuda", dtype=dtype).uniform_(0.0, 7.0),
            "N": 1000
        })

        test_cases.append({
            "input": torch.empty(10000, device="cuda", dtype=dtype).uniform_(0.0, 1.0),
            "N": 10000
        })

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 25000000
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "N": N
        }
