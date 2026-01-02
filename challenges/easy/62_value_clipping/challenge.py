import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Clipping Kernel", atol=1e-05, rtol=1e-05, num_gpus=1, access_tier="free"
        )

    def reference_impl(
        self, input: torch.Tensor, output: torch.Tensor, N: int, lo: float, hi: float
    ):
        assert input.shape == (N,)
        assert output.shape == (N,)
        assert input.dtype == output.dtype
        assert input.device == output.device
        output.copy_(input.clamp(min=lo, max=hi))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "lo": (ctypes.c_float, "in"),
            "hi": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        input = torch.tensor([1.5, -2.0, 3.0, 4.5], device="cuda", dtype=dtype)
        output = torch.empty(N, device="cuda", dtype=dtype)
        lo, hi = 0.0, 3.5
        return {"input": input, "output": output, "N": N, "lo": lo, "hi": hi}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Example 2
        N = 3
        tests.append(
            {
                "input": torch.tensor([-1.0, 2.0, 5.0], device="cuda", dtype=dtype),
                "output": torch.empty(N, device="cuda", dtype=dtype),
                "N": N,
                "lo": -0.5,
                "hi": 2.5,
            }
        )

        # all zeros
        N = 42
        tests.append(
            {
                "input": torch.zeros(N, device="cuda", dtype=dtype),
                "output": torch.empty(N, device="cuda", dtype=dtype),
                "N": N,
                "lo": -1.0,
                "hi": 1.0,
            }
        )

        # negative numbers
        N = 6
        tests.append(
            {
                "input": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], device="cuda", dtype=dtype),
                "output": torch.empty(N, device="cuda", dtype=dtype),
                "N": N,
                "lo": -3.0,
                "hi": -1.0,
            }
        )

        # mixed positive/negative
        N = 4
        tests.append(
            {
                "input": torch.tensor([-0.5, 0.0, -1.5, 1.0], device="cuda", dtype=dtype),
                "output": torch.empty(N, device="cuda", dtype=dtype),
                "N": N,
                "lo": -1.0,
                "hi": 0.5,
            }
        )

        # large values
        N = 1024
        tests.append(
            {
                "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
                "output": torch.empty(N, device="cuda", dtype=dtype),
                "N": N,
                "lo": -50.9,
                "hi": 50.1,
            }
        )

        # large N
        N = 2048
        tests.append(
            {
                "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-50.0, 50.0),
                "output": torch.empty(N, device="cuda", dtype=dtype),
                "N": N,
                "lo": -25.5,
                "hi": 25.05,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 100000
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N,
            "lo": -51.24,
            "hi": 39.51,
        }
