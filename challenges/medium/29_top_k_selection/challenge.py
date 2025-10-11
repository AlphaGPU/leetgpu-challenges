import ctypes
import torch
from typing import Any, List, Dict
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Top K Selection",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, k: int):
        assert input.shape == (N,)
        assert output.shape == (k,)
        assert input.dtype == output.dtype == torch.float32
        assert input.device == output.device
        topk = torch.topk(input, k, largest=True).values
        output.copy_(topk)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "k": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], device="cuda", dtype=dtype)
        output = torch.empty(3, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "k": 3
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append({
            "input": torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], device="cuda", dtype=dtype),
            "output": torch.empty(3, device="cuda", dtype=dtype),
            "N": 5,
            "k": 3
        })
        # negative_numbers
        tests.append({
            "input": torch.tensor([-2.0, -1.0, -3.0, -4.0, -5.0, -6.0], device="cuda", dtype=dtype),
            "output": torch.empty(2, device="cuda", dtype=dtype),
            "N": 6,
            "k": 2
        })
        # all_equal
        tests.append({
            "input": torch.tensor([7.0, 7.0, 7.0, 7.0], device="cuda", dtype=dtype),
            "output": torch.empty(3, device="cuda", dtype=dtype),
            "N": 4,
            "k": 3
        })
        # single_element
        tests.append({
            "input": torch.tensor([42.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1,
            "k": 1
        })
        # reverse_sorted
        tests.append({
            "input": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device="cuda", dtype=dtype),
            "output": torch.empty(2, device="cuda", dtype=dtype),
            "N": 5,
            "k": 2
        })
        # large_random (simulated; actual is random in runner)
        N, k = 1000, 10
        tests.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "output": torch.empty(k, device="cuda", dtype=dtype),
            "N": N,
            "k": k
        })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000000
        k = 100
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1e6, 1e6),
            "output": torch.empty(k, device="cuda", dtype=dtype),
            "N": N,
            "k": k
        } 