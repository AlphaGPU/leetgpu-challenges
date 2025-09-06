import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Prime Checker",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, n: int, output: torch.Tensor):
        # Validate input types and shapes
        assert output.shape == (1,)
        assert output.dtype == torch.int32

        if n == 1:
            output[0] = 0
        elif n == 2:
            output[0] = 1
        else:
            output[0] = 1
            for i in range(2, n):
                if (n % i) == 0:
                    output[0] = 0
                    break
                if i * i > n:
                    break

    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "n": ctypes.c_int,
            "output": ctypes.POINTER(ctypes.c_int)
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "n": 2,
            "output": output
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append({
            "n": 9,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        tests.append({
            "n": 113,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        tests.append({
            "n": 1,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        # small primes
        tests.append({
            "n": 10007,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        tests.append({
            "n": 99991,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        # small no primes
        tests.append({
            "n": 10011,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        tests.append({
            "n": 9995,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        # medium primes
        tests.append({
            "n": 999983,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        tests.append({
            "n": 1000003,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        # medium no primes
        tests.append({
            "n": 999987,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        tests.append({
            "n": 1000009,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        # large primes
        tests.append({
            "n": 49999991,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        tests.append({
            "n": 50000017,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        # large no primes
        tests.append({
            "n": 50000000,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        tests.append({
            "n": 100000000,
            "output": torch.empty(1, device="cuda", dtype=dtype)
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "n": 99999989,
            "output": output
        }