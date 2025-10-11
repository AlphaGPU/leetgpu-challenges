import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Count 3D Array Element",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int, P: int):
        # Validate input types and shapes
        assert input.shape == (N, M, K)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # count the number of element with value k in an input array
        equality_tensor = (input == P)
        output[0] = torch.sum(equality_tensor)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "M": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"), 
            "P": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        
        input = torch.tensor([[[1, 2, 3], [4, 5, 1]], [[1, 1, 1], [2, 2, 2]]], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 2,
            "M": 2,
            "K": 3,
            "P": 1
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([[[1, 2, 3], [4, 5, 1]], [[1, 1, 1], [2, 2, 2]]], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 2,
            "M": 2,
            "K": 3,
            "P": 1
        })

        # all_same_value
        tests.append({
            "input": torch.tensor([[[2]*16] * 3]*15, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 15,
            "M": 3,
            "K": 16,
            "P": 2
        })

        # increasing_sequence
        tests.append({
            "input": torch.randint(1, 11, (50, 50, 50), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 50,
            "M": 50,
            "K": 50,
            "P": 5
        })

        # medium_size
        tests.append({
            "input": torch.randint(1, 101, (100, 100, 100), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 100,
            "M": 100,
            "K": 100,
            "P": 51
        })

        # large_size
        tests.append({
            "input": torch.randint(1, 11, (100, 200, 300), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 100,
            "M": 200,
            "K": 300,
            "P": 3
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(1, 3, (500, 500, 500), device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 500,
            "M": 500,
            "K": 500,
            "P": 2
        }
