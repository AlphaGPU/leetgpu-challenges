import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Hadamard Transform"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        B: int,
        N: int,
    ):
        assert input.shape == output.shape == (B, N)
        assert input.dtype == output.dtype == torch.float32
        assert N >= 1 and (N & (N - 1)) == 0

        x = input.clone()
        h = 1
        while h < N:
            x_view = x.view(B, N // (2 * h), 2, h)
            a = x_view[:, :, 0, :].clone()
            b = x_view[:, :, 1, :].clone()
            x_view[:, :, 0, :] = a + b
            x_view[:, :, 1, :] = a - b
            h *= 2
        output.copy_(x / math.sqrt(N))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        B, N = 1, 4
        input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device, dtype=dtype)
        output = torch.empty((B, N), device=self.device, dtype=dtype)
        return {"input": input, "output": output, "B": B, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests: List[Dict[str, Any]] = []

        def make_case(B, N, values=None, low=-1.0, high=1.0):
            if values is None:
                inp = torch.empty((B, N), device=self.device, dtype=dtype).uniform_(low, high)
            else:
                inp = torch.tensor(values, device=self.device, dtype=dtype).view(B, N)
            out = torch.empty((B, N), device=self.device, dtype=dtype)
            return {"input": inp, "output": out, "B": B, "N": N}

        # Trivial: N=1 — Hadamard of a single value is itself.
        tests.append(make_case(3, 1, values=[[1.5], [-2.0], [0.0]]))

        # Smallest non-trivial transform.
        tests.append(make_case(1, 2, values=[[1.0, -1.0]]))

        # Example-sized case with explicit values for easy checking.
        tests.append(make_case(2, 4, values=[[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, -0.5, 1.0]]))

        # All zeros — output must also be zero.
        tests.append(make_case(4, 16, values=[[0.0] * 16] * 4))

        # Constant rows — only the first transform coefficient is non-zero.
        tests.append(make_case(2, 8, values=[[2.5] * 8, [-3.0] * 8]))

        # Power-of-two sizes with random values, mixed signs.
        tests.append(make_case(8, 32, low=-1.0, high=1.0))
        tests.append(make_case(32, 128, low=-2.0, high=2.0))

        # Large rows.
        tests.append(make_case(4, 1024, low=-1.0, high=1.0))

        # Wide batch, medium row.
        tests.append(make_case(1000, 64, low=-5.0, high=5.0))

        # Stress: realistic size.
        tests.append(make_case(256, 2048, low=-1.0, high=1.0))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        B, N = 8192, 4096
        input = torch.empty((B, N), device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        output = torch.empty((B, N), device=self.device, dtype=dtype)
        return {"input": input, "output": output, "B": B, "N": N}
