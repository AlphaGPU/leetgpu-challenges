import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="2D Subarray Sum",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, M: int, S_ROW: int, E_ROW: int, S_COL: int, E_COL: int):
        # Validate input types and shapes
        assert input.shape == (N, M)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # add all elements of subarray (input[S_ROW..E_ROW][S_COL..E_COL])
        output[0] = torch.sum(input[S_ROW:E_ROW+1, S_COL:E_COL+1]);

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "M": (ctypes.c_int, "in"),
            "S_ROW": (ctypes.c_int, "in"),
            "E_ROW": (ctypes.c_int, "in"),
            "S_COL": (ctypes.c_int, "in"),
            "E_COL": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([[1, 2, 3], [4, 5, 1]], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 2,
            "M": 3,
            "S_ROW": 0,
            "E_ROW": 1,
            "S_COL": 1,
            "E_COL": 2
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([[5, 10], [5, 2]], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 2,
            "M": 2,
            "S_ROW": 0,
            "E_ROW": 0,
            "S_COL": 1,
            "E_COL": 1
        })

        # all_same_value
        tests.append({
            "input": torch.tensor([[2]*16] * 3, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 3,
            "M": 16,
            "S_ROW": 0,
            "E_ROW": 2,
            "S_COL": 0,
            "E_COL": 15
        })

        # increasing_sequence
        tests.append({
            "input": torch.randint(1, 11, (50,50), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 50,
            "M": 50,
            "S_ROW": 0,
            "E_ROW": 49,
            "S_COL": 0,
            "E_COL": 49
        })

        # medium_size
        tests.append({
            "input": torch.randint(1, 11, (100,100), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 100,
            "M": 100,
            "S_ROW": 0,
            "E_ROW": 79,
            "S_COL": 1,
            "E_COL": 87
        })

        # large_size
        tests.append({
            "input": torch.randint(1, 11, (1000,1000), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1000,
            "M": 1000,
            "S_ROW": 10,
            "E_ROW": 951,
            "S_COL": 12,
            "E_COL": 810
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(1, 11, (10000,10000), device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 10000,
            "M": 10000,
            "S_ROW": 0,
            "E_ROW": 9998,
            "S_COL": 1,
            "E_COL": 9999
        }