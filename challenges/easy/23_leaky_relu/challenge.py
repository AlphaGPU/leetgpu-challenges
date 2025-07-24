import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Leaky ReLU",
            atol=1e-06,
            rtol=1e-06,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert input.shape == (N,)
        assert output.shape == (N,)
        assert input.dtype == output.dtype
        assert input.device == output.device

        # Apply Leaky ReLU: f(x) = x if x > 0, else 0.01 * x
        alpha = 0.01
        output[:] = torch.where(input > 0, input, alpha * input)

    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "input": ctypes.POINTER(ctypes.c_float),
            "output": ctypes.POINTER(ctypes.c_float),
            "N": ctypes.c_int
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_tensor = torch.tensor([1.0, -2.0, 3.0, -4.0], device="cuda", dtype=dtype)
        output_tensor = torch.empty(4, device="cuda", dtype=dtype)
        return {
            "input": input_tensor,
            "output": output_tensor,
            "N": 4
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []

        # basic_example
        test_cases.append({
            "input": torch.tensor([1.0, -2.0, 3.0, -4.0], device="cuda", dtype=dtype),
            "output": torch.zeros(4, device="cuda", dtype=dtype),
            "N": 4
        })

        # all_positive
        test_cases.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 5
        })

        # all_negative
        test_cases.append({
            "input": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 5
        })

        # zeros
        test_cases.append({
            "input": torch.zeros(1024, device="cuda", dtype=dtype),
            "output": torch.zeros(1024, device="cuda", dtype=dtype),
            "N": 1024
        })

        # medium_random
        test_cases.append({
            "input": torch.empty(10000, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "output": torch.zeros(10000, device="cuda", dtype=dtype),
            "N": 10000
        })

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000000  # Large vector for performance testing
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "output": torch.zeros(N, device="cuda", dtype=dtype),
            "N": N
        }
