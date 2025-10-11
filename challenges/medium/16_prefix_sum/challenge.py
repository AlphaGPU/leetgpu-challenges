import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Prefix Sum",
            atol=1e-02,
            rtol=1e-02,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert input.shape == (N,)
        assert output.shape == (N,)
        result = torch.cumsum(input, dim=0)
        output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        output = torch.empty(4, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 4
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype),
            "output": torch.empty(4, device="cuda", dtype=dtype),
            "N": 4
        })
        # mixed_signs
        tests.append({
            "input": torch.tensor([5.0, -2.0, 3.0, 1.0, -4.0], device="cuda", dtype=dtype),
            "output": torch.empty(5, device="cuda", dtype=dtype),
            "N": 5
        })
        # single_element
        tests.append({
            "input": torch.tensor([42.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1
        })
        # power_of_two
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "output": torch.empty(8, device="cuda", dtype=dtype),
            "N": 8
        })
        # all_zeros
        tests.append({
            "input": torch.empty(1024, device="cuda", dtype=dtype).zero_(),
            "output": torch.empty(1024, device="cuda", dtype=dtype),
            "N": 1024
        })
        # random_large
        tests.append({
            "input": torch.empty(2025, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "output": torch.empty(2025, device="cuda", dtype=dtype),
            "N": 2025
        })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 250000
        input = torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0)
        output = torch.empty(N, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": N
        } 