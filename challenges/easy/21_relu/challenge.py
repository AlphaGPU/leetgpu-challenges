import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="ReLU",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert input.shape == (N,)
        assert output.shape == (N,)
        assert input.dtype == output.dtype
        assert input.device == output.device

        # Apply ReLU: max(0, x)
        output.copy_(torch.relu(input))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="cuda", dtype=dtype)
        output_tensor = torch.empty(5, device="cuda", dtype=dtype)
        return {
            "input": input_tensor,
            "output": output_tensor,
            "N": 5
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32

        test_cases = []

        # Fixed-value test cases
        test_cases.append({
            "input": torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 5
        })

        test_cases.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 5
        })

        test_cases.append({
            "input": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 5
        })

        test_cases.append({
            "input": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 5
        })

        test_cases.append({
            "input": torch.tensor([-1000.0, -100.0, 0.0, 100.0, 1000.0], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 5
        })

        test_cases.append({
            "input": torch.tensor([-0.001, -0.0001, 0.0, 0.0001, 0.001], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 5
        })

        # Random range test cases
        test_cases.append({
            "input": torch.empty(1024, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "output": torch.zeros(1024, device="cuda", dtype=dtype),
            "N": 1024
        })

        test_cases.append({
            "input": torch.empty(10000, device="cuda", dtype=dtype).uniform_(-50.0, 50.0),
            "output": torch.zeros(10000, device="cuda", dtype=dtype),
            "N": 10000
        })

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 25000000  # Large vector for performance testing
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "output": torch.zeros(N, device="cuda", dtype=dtype),
            "N": N
        }
