import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Gaussian Error Gated Linear Unit",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert N % 2 == 0
        assert input.shape == (N,)
        assert output.shape == (N // 2,)
        assert input.dtype == output.dtype
        assert input.device == output.device

        x1, x2 = input.chunk(2)
        gelu = 0.5 * x2 * (1.0 + torch.erf(x2 / math.sqrt(2.0)))
        output.copy_(x1 * gelu)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 2
        input = torch.tensor([1.0, 1.0], device="cuda", dtype=dtype)
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
        N = 4
        tests.append(
            {
                "input": torch.tensor([2.0, -1.0, 1.0, 0.5], device="cuda", dtype=dtype),
                "output": torch.empty(N // 2, device="cuda", dtype=dtype),
                "N": N,
            }
        )

        # all zeros
        N = 42
        tests.append(
            {
                "input": torch.zeros(N, device="cuda", dtype=dtype),
                "output": torch.empty(N // 2, device="cuda", dtype=dtype),
                "N": N,
            }
        )

        # negative numbers
        N = 6
        tests.append(
            {
                "input": torch.tensor(
                    [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], device="cuda", dtype=dtype
                ),
                "output": torch.empty(N // 2, device="cuda", dtype=dtype),
                "N": N,
            }
        )

        # mixed positive/negative
        N = 4
        tests.append(
            {
                "input": torch.tensor([-0.5, 0.0, -1.5, 1.0], device="cuda", dtype=dtype),
                "output": torch.empty(N // 2, device="cuda", dtype=dtype),
                "N": N,
            }
        )

        # large values
        N = 1024
        tests.append(
            {
                "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
                "output": torch.empty(N // 2, device="cuda", dtype=dtype),
                "N": N,
            }
        )

        # large N
        N = 100000
        tests.append(
            {
                "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-50.0, 50.0),
                "output": torch.empty(N // 2, device="cuda", dtype=dtype),
                "N": N,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 1000000
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N,
        }
