import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Sigmoid Linear Unit",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert input.shape == output.shape == (N,)
        assert input.dtype == output.dtype
        assert input.device == output.device
        
        # Scale and shift
        output.copy_(input * torch.sigmoid(input))
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        input = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        output = torch.empty(N, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_small
        N = 3
        tests.append({
            "input": torch.tensor([0.5, 1.0, -0.5], device="cuda", dtype=dtype),
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N
        })

        # single_element
        N = 1
        tests.append({
            "input": torch.tensor([2.0], device="cuda", dtype=dtype),
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N
        })

        # all zeros
        N = 42
        tests.append({
            "input": torch.zeros(N, device="cuda", dtype=dtype),
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N
        })

        # negative numbers
        N = 5
        tests.append({
            "input": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], device="cuda", dtype=dtype),
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N
        })

        # mixed positive/negative
        N = 4
        tests.append({
            "input": torch.tensor([-0.5, 0.0, 0.5, 1.0], device="cuda", dtype=dtype),
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N
        })

        # large values
        N = 1024
        tests.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N
        })

        # large N
        N = 10000
        tests.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-50.0, 50.0),
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-50.0, 50.0),
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N
        }