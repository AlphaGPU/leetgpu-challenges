import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Histogramming",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
        # Validate input types and shapes
        assert input.dtype == torch.int32
        assert histogram.dtype == torch.int32
        assert input.numel() == N
        assert histogram.numel() == num_bins
        # Zero out the histogram
        histogram.zero_()
        # Only count valid input values
        valid_mask = (input >= 0) & (input < num_bins)
        valid_input = input[valid_mask]
        counts = torch.bincount(valid_input, minlength=num_bins)
        histogram.copy_(counts)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "histogram": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "num_bins": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([0, 1, 2, 1, 0], device="cuda", dtype=dtype)
        histogram = torch.empty(3, device="cuda", dtype=dtype)
        return {
            "input": input,
            "histogram": histogram,
            "N": 5,
            "num_bins": 3
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([0, 1, 2, 1, 0], device="cuda", dtype=dtype),
            "histogram": torch.zeros(3, device="cuda", dtype=dtype),
            "N": 5,
            "num_bins": 3
        })

        # all_same_value
        tests.append({
            "input": torch.tensor([2]*16, device="cuda", dtype=dtype),
            "histogram": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 16,
            "num_bins": 5
        })

        # increasing_sequence
        tests.append({
            "input": torch.randint(0, 4, (32,), device="cuda", dtype=dtype),
            "histogram": torch.zeros(4, device="cuda", dtype=dtype),
            "N": 32,
            "num_bins": 4
        })

        # medium_size
        tests.append({
            "input": torch.randint(0, 10, (1000,), device="cuda", dtype=dtype),
            "histogram": torch.zeros(10, device="cuda", dtype=dtype),
            "N": 1000,
            "num_bins": 10
        })

        # large_multi_block
        tests.append({
            "input": torch.randint(0, 128, (10000,), device="cuda", dtype=dtype),
            "histogram": torch.zeros(128, device="cuda", dtype=dtype),
            "N": 10000,
            "num_bins": 128
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(0, 256, (50000000,), device="cuda", dtype=dtype)
        histogram = torch.zeros(256, device="cuda", dtype=dtype)
        return {
            "input": input,
            "histogram": histogram,
            "N": 50000000,
            "num_bins": 256
        }
