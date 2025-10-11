import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Rainbow Table",
            atol=0,
            rtol=0,
            num_gpus=1,
            access_tier="free"
        )

    def fnv1a_hash(self, x: torch.Tensor) -> torch.Tensor:
        FNV_PRIME = 16777619
        OFFSET_BASIS = 2166136261
        x_int = x.to(torch.int64)
        hash_val = torch.full_like(x_int, OFFSET_BASIS, dtype=torch.int64)
        for byte_pos in range(4):
            byte = (x_int >> (byte_pos * 8)) & 0xFF
            hash_val = (hash_val ^ byte) * FNV_PRIME
            hash_val = hash_val & 0xFFFFFFFF
        return hash_val

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, R: int):
        assert input.shape == (N,)
        assert output.shape == (N,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.uint32

        current = input

        # Apply hash R times
        for _ in range(R):
            current = self.fnv1a_hash(current)

        # Reinterpret the lower 32 bits as uint32
        output.copy_(current.to(torch.int32).view(torch.uint32))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_int32), "in"),
            "output": (ctypes.POINTER(ctypes.c_uint32), "out"),
            "N": (ctypes.c_int, "in"),
            "R": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        input_tensor = torch.tensor([123, 456, 789], device="cuda", dtype=torch.int32)
        output_tensor = torch.empty(3, device="cuda", dtype=torch.uint32)
        return {
            "input": input_tensor,
            "output": output_tensor,
            "N": 3,
            "R": 2
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32

        test_cases = []

        # basic_example
        test_cases.append({
            "input": torch.tensor([123, 456, 789], device="cuda", dtype=dtype),
            "output": torch.zeros(3, device="cuda", dtype=torch.uint32),
            "N": 3,
            "R": 2
        })

        # zero_and_max
        test_cases.append({
            "input": torch.tensor([0, 1, 2147483647], device="cuda", dtype=dtype),
            "output": torch.zeros(3, device="cuda", dtype=torch.uint32),
            "N": 3,
            "R": 3
        })

        # single_round
        test_cases.append({
            "input": torch.tensor([1, 2, 3, 4, 5], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=torch.uint32),
            "N": 5,
            "R": 1
        })

        # many_rounds
        test_cases.append({
            "input": torch.randint(0, 2147483647 + 1, (1024,), device="cuda", dtype=dtype),
            "output": torch.zeros(1024, device="cuda", dtype=torch.uint32),
            "N": 1024,
            "R": 50
        })

        # large_size
        test_cases.append({
            "input": torch.randint(0, 2147483647 + 1, (10000,), device="cuda", dtype=dtype),
            "output": torch.zeros(10000, device="cuda", dtype=torch.uint32),
            "N": 10000,
            "R": 10
        })

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        N, R = 5000000, 10  # Large array with moderate rounds for performance testing
        return {
            "input": torch.randint(0, 2147483647 + 1, (N,), device="cuda", dtype=torch.int32),
            "output": torch.zeros(N, device="cuda", dtype=torch.uint32),
            "N": N,
            "R": R
        }
