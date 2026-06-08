import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Layer Normalization", atol=1e-04, rtol=1e-04, num_gpus=1, access_tier="free"
        )

    def reference_impl(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
        N: int,
        C: int,
        eps: float,
    ):
        assert input.shape == output.shape == (N, C)
        assert weight.shape == bias.shape == (C,)
        assert input.dtype == weight.dtype == bias.dtype == output.dtype
        assert input.device == weight.device == bias.device == output.device
        assert str(input.device).startswith("cuda")

        mean = input.mean(dim=1, keepdim=True)
        var = input.var(dim=1, keepdim=True, unbiased=False)
        normalized = (input - mean) / torch.sqrt(var + eps)
        output.copy_(weight * normalized + bias)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "bias": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, C = 2, 4
        input = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 0.0, 1.0]], device="cuda", dtype=dtype
        )
        weight = torch.ones(C, device="cuda", dtype=dtype)
        bias = torch.zeros(C, device="cuda", dtype=dtype)
        output = torch.empty((N, C), device="cuda", dtype=dtype)
        eps = 1e-5
        return {
            "input": input,
            "weight": weight,
            "bias": bias,
            "output": output,
            "N": N,
            "C": C,
            "eps": eps,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # edge: single element per row
        N, C = 1, 1
        tests.append(
            {
                "input": torch.tensor([[3.0]], device="cuda", dtype=dtype),
                "weight": torch.tensor([1.0], device="cuda", dtype=dtype),
                "bias": torch.tensor([0.5], device="cuda", dtype=dtype),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # edge: 2x2, all zeros
        N, C = 2, 2
        tests.append(
            {
                "input": torch.zeros((N, C), device="cuda", dtype=dtype),
                "weight": torch.ones(C, device="cuda", dtype=dtype),
                "bias": torch.zeros(C, device="cuda", dtype=dtype),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # edge: 4x4, negative values
        N, C = 4, 4
        tests.append(
            {
                "input": torch.tensor(
                    [
                        [-1.0, -2.0, -3.0, -4.0],
                        [1.0, 2.0, 3.0, 4.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [-2.0, 0.0, 2.0, 4.0],
                    ],
                    device="cuda",
                    dtype=dtype,
                ),
                "weight": torch.tensor([1.0, 2.0, 1.0, 0.5], device="cuda", dtype=dtype),
                "bias": torch.tensor([0.0, 0.0, 1.0, -1.0], device="cuda", dtype=dtype),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # power-of-2: 8x16
        N, C = 8, 16
        tests.append(
            {
                "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "weight": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.5, 2.0),
                "bias": torch.empty(C, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # power-of-2: 32x64
        N, C = 32, 64
        tests.append(
            {
                "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "weight": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.5, 2.0),
                "bias": torch.empty(C, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # power-of-2: 128x256
        N, C = 128, 256
        tests.append(
            {
                "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "weight": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.5, 2.0),
                "bias": torch.empty(C, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # non-power-of-2: 7x30
        N, C = 7, 30
        tests.append(
            {
                "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "weight": torch.ones(C, device="cuda", dtype=dtype),
                "bias": torch.zeros(C, device="cuda", dtype=dtype),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # non-power-of-2: 15x100
        N, C = 15, 100
        tests.append(
            {
                "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
                "weight": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.1, 3.0),
                "bias": torch.empty(C, device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # non-power-of-2: 25x255
        N, C = 25, 255
        tests.append(
            {
                "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "weight": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.5, 2.0),
                "bias": torch.empty(C, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        # realistic: 512x768 (BERT hidden size)
        N, C = 512, 768
        tests.append(
            {
                "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "weight": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.5, 2.0),
                "bias": torch.empty(C, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty((N, C), device="cuda", dtype=dtype),
                "N": N,
                "C": C,
                "eps": 1e-5,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, C = 65536, 512
        return {
            "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-5.0, 10.0),
            "weight": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.5, 2.0),
            "bias": torch.empty(C, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N,
            "C": C,
            "eps": 1e-5,
        }
