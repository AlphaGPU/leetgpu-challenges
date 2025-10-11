import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Max Subarray Sum",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
        # Validate input types and shapes
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # Computes the maximum sum of any contiguous subarray of length exactly window_size
        # using a sliding window approach.

        # Compute the sum of the first window_size elements (the initial window)
        current_sum = input[:window_size].sum()

        # Initialize max_sum with the sum of the first window
        max_sum = current_sum

        # Slide the window across the array from index window_size to N - 1
        for i in range(window_size, N):
            # Update the current sum by subtracting the element leaving the window
            # and adding the new element entering the window
            current_sum += input[i] - input[i - window_size]
    
            # Update max_sum if the current sum is greater
            max_sum = torch.max(max_sum, current_sum)

        # Store the final result in the output tensor
        output[0] = max_sum

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "window_size": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([1, 2, 4, 2, 3], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "window_size": 2
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([-1, -4, -2, 1], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4,
            "window_size": 3
        })

        # all_same_value
        tests.append({
            "input": torch.tensor([2]*16, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 16,
            "window_size": 15
        })

        # all_minus_value
        tests.append({
            "input": torch.tensor([-10]*1000, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1000,
            "window_size": 500
        })

        # increasing_sequence
        tests.append({
            "input": torch.randint(-10, 11, (123,), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 123,
            "window_size": 7
        })

        # medium_size
        tests.append({
            "input": torch.randint(-10, 11, (1000,), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1000,
            "window_size": 476
        })

        # large_size
        tests.append({
            "input": torch.randint(-10, 11, (10000,), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 10000,
            "window_size": 7011
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(-10, 11, (50000,), device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 50000,
            "window_size": 25000
        }