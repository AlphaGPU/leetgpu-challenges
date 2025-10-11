import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Reduction",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == output.dtype
        assert input.device == output.device
        output[0] = torch.sum(input)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        N = 8
        return {"input": input, "output": output, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 8
        })
        # negative_numbers
        tests.append({
            "input": torch.tensor([-2.5, 1.5, -1.0, 2.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4
        })
        # single_element
        tests.append({
            "input": torch.tensor([42.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1
        })
        # all_zeros
        tests.append({
            "input": torch.zeros(1024, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1024
        })
        # all_ones
        tests.append({
            "input": torch.ones(1024, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1024
        })
        # non_power_of_two
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 5
        })
        # large_random
        tests.append({
            "input": torch.empty(10000, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 10000
        })
        # large_random_2
        tests.append({
            "input": torch.empty(15000000, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 15000000
        })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000000
        input = torch.empty(N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {"input": input, "output": output, "N": N} 