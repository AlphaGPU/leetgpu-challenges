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

        # 2D inclusive prefix sum, padded with a zero row and column
        psum = input.cumsum(dim=0).cumsum(dim=1)
        padded = torch.zeros((N + 1, N + 1), dtype=torch.int32, device=input.device)
        padded[1:, 1:] = psum

        # Sum of every window_size x window_size window via the inclusion-exclusion identity
        top_left = padded[:-window_size, :-window_size]
        top_right = padded[:-window_size, window_size:]
        bottom_left = padded[window_size:, :-window_size]
        bottom_right = padded[window_size:, window_size:]
        window_sums = bottom_right - top_right - bottom_left + top_left

        output[0] = torch.max(window_sums)

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

        # all_negative_values
        tests.append({
            "input": torch.tensor(
                [[-1, -2, -3], [-4, -5, -1], [-5, -1, -7]], device="cuda", dtype=dtype
            ),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 3,
            "window_size": 2
        })

        # single_element
        tests.append({
            "input": torch.tensor([[-7]], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1,
            "window_size": 1
        })

        # all_zeros
        tests.append({
            "input": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 2,
            "window_size": 2
        })

        # mixed_signs_small
        tests.append({
            "input": torch.tensor(
                [[0, -3, 4, 1], [2, 5, -6, 0], [-1, 7, 2, -8], [3, 0, -2, 6]],
                device="cuda",
                dtype=dtype
            ),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4,
            "window_size": 3
        })

        # all_same_value, window covers the whole matrix
        tests.append({
            "input": torch.full((16, 16), 2, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 16,
            "window_size": 16
        })

        # window_size of 1 reduces to a plain maximum
        tests.append({
            "input": torch.randint(-10, 11, (32, 32), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 32,
            "window_size": 1
        })

        # all_negative_values, non-power-of-2 size
        tests.append({
            "input": torch.randint(-10, 0, (100, 100), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 100,
            "window_size": 37
        })

        # mixed values, non-power-of-2 size
        tests.append({
            "input": torch.randint(-10, 11, (255, 255), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 255,
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
