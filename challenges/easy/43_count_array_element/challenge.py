import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Count Array Element",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, K: int):
        # Validate input types and shapes
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # count the number of element with value k in an input array
        equality_tensor = (input == K)
        output[0] = torch.sum(equality_tensor)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([1, 2, 3, 4, 1], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "K": 1
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([1, 2, 3, 4, 1], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 5,
            "K": 1
        })

        # all_same_value
        tests.append({
            "input": torch.tensor([2]*16, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 16,
            "K": 2
        })

        # increasing_sequence
        tests.append({
            "input": torch.randint(1, 5, (32,), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 32,
            "K": 4
        })

        # medium_size
        tests.append({
            "input": torch.randint(1, 10, (1000,), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1000,
            "K": 5
        })

        # large_size
        tests.append({
            "input": torch.randint(1, 1000, (100000,), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 100000,
            "K": 501
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(1, 100001, (100000000,), device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 100000000,
            "K": 501010
        }
