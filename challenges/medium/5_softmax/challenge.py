import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Softmax",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert input.shape == output.shape == (N,)
        assert input.dtype == output.dtype
        assert input.device == output.device
        max_val = torch.max(input)
        exp_x = torch.exp(input - max_val)
        sum_exp = torch.sum(exp_x)
        output.copy_(exp_x / sum_exp)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype)
        output = torch.empty(3, device="cuda", dtype=dtype)
        N = 3
        return {"input": input, "output": output, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_small
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype),
            "output": torch.empty(3, device="cuda", dtype=dtype),
            "N": 3
        })
        # all_zeros
        tests.append({
            "input": torch.tensor([0.0, 0.0, 0.0, 0.0], device="cuda", dtype=dtype),
            "output": torch.empty(4, device="cuda", dtype=dtype),
            "N": 4
        })
        # negative_numbers
        tests.append({
            "input": torch.tensor([-1.0, -2.0, -3.0], device="cuda", dtype=dtype),
            "output": torch.empty(3, device="cuda", dtype=dtype),
            "N": 3
        })
        # mixed_positive_negative
        tests.append({
            "input": torch.tensor([1.0, -2.0, 3.0, -4.0], device="cuda", dtype=dtype),
            "output": torch.empty(4, device="cuda", dtype=dtype),
            "N": 4
        })
        # very_small_numbers
        tests.append({
            "input": torch.tensor([1e-6, 1e-7, 1e-8, 1e-9], device="cuda", dtype=dtype),
            "output": torch.empty(4, device="cuda", dtype=dtype),
            "N": 4
        })
        # large_numbers
        tests.append({
            "input": torch.tensor([10.0, 15.0, 20.0], device="cuda", dtype=dtype),
            "output": torch.empty(3, device="cuda", dtype=dtype),
            "N": 3
        })
        # single_element
        tests.append({
            "input": torch.tensor([5.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1
        })
        # all_same_values
        tests.append({
            "input": torch.tensor([2.5] * 10, device="cuda", dtype=dtype),
            "output": torch.empty(10, device="cuda", dtype=dtype),
            "N": 10
        })
        # large_array
        tests.append({
            "input": torch.empty(2048, device="cuda", dtype=dtype).uniform_(0.0, 10.0),
            "output": torch.empty(2048, device="cuda", dtype=dtype),
            "N": 2048
        })
        # large_max_small_values
        tests.append({
            "input": torch.tensor([1000.0, 1.0, 2.0, 3.0], device="cuda", dtype=dtype),
            "output": torch.empty(4, device="cuda", dtype=dtype),
            "N": 4
        })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 500000
        input = torch.empty(N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0)
        output = torch.empty(N, device="cuda", dtype=dtype)
        return {"input": input, "output": output, "N": N} 