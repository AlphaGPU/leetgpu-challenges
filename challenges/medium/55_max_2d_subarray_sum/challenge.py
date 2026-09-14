import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Max 2D Subarray Sum",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
        # Validate input types and shapes
        assert input.shape == (N, N)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        psum = input.cumsum(dim=0).cumsum(dim=1)
        padded = torch.zeros((N+1, N+1), dtype=torch.int32)
        padded[1:, 1:] = psum

        top_left = padded[:-window_size, :-window_size]
        top_right = padded[:-window_size, window_size:]
        bottom_left = padded[window_size:, :-window_size]
        bottom_right = padded[window_size:, window_size:]
        window_sums = bottom_right - top_right - bottom_left + top_left

        max_sum = torch.max(window_sums)
        output[0] = max_sum

    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "input": ctypes.POINTER(ctypes.c_int),
            "output": ctypes.POINTER(ctypes.c_int),
            "N": ctypes.c_int,
            "window_size": ctypes.c_int
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([[1, 2, 3], [4, 5, 1], [5, 1, 7]], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 3,
            "window_size": 2
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([[-1, -2, -3], [-4, -5, -1], [-5, -1, -7]], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 3,
            "window_size": 2
        })

        # all_same_value
        tests.append({
            "input": torch.tensor([[2]*16] * 16, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 16,
            "window_size": 16
        })

        tests.append({
            "input": torch.tensor([[2]*16] * 16, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 16,
            "window_size": 15
        })

        tests.append({
            "input": torch.tensor([[2]*16] * 16, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 16,
            "window_size": 1
        })

        # all_minus_value
        tests.append({
            "input": torch.tensor([[-10]*10]*10, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 10,
            "window_size": 5
        })

        tests.append({
            "input": torch.randint(-10, 0, (123, 123), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 123,
            "window_size": 7
        })

        # increasing_sequence
        tests.append({
            "input": torch.randint(-10, 11, (123, 123), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 123,
            "window_size": 7
        })

        # medium_size
        tests.append({
            "input": torch.randint(-10, 11, (1000, 1000), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1000,
            "window_size": 476
        })

        # large_size
        tests.append({
            "input": torch.randint(-10, 11, (3000, 3000), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 3000,
            "window_size": 2011
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(-10, 11, (5000, 5000), device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5000,
            "window_size": 2500
        }