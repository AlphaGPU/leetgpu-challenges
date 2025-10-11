import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Swish-Gated Linear Unit",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert N % 2 == 0
        assert input.shape ==  (N,)
        assert output.shape ==  (N//2,)
        assert input.dtype == output.dtype
        assert input.device == output.device

        x1, x2 = input.chunk(2)
        output.copy_((x1 * torch.sigmoid(x1)) * x2)
        
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
        output = torch.empty(N // 2, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_small
        N = 2
        tests.append({
            "input": torch.tensor([0.5, 1.0], device="cuda", dtype=dtype),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # all zeros
        N = 42
        tests.append({
            "input": torch.zeros(N, device="cuda", dtype=dtype),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # negative numbers
        N = 6
        tests.append({
            "input": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], device="cuda", dtype=dtype),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # mixed positive/negative
        N = 4
        tests.append({
            "input": torch.tensor([-0.5, 0.0, - 1.5, 1.0], device="cuda", dtype=dtype),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # large values
        N = 1024
        tests.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # large N
        N = 2048
        tests.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-50.0, 50.0),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 100000
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        }