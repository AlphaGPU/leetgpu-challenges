import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Gaussian Blur",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
        input_2d = input.view(1, 1, input_rows, input_cols)
        kernel_2d = kernel.view(1, 1, kernel_rows, kernel_cols)
        pad_h = kernel_rows // 2
        pad_w = kernel_cols // 2
        result = torch.nn.functional.conv2d(input_2d, kernel_2d, padding=(pad_h, pad_w))
        output[:] = result.view(-1)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "kernel": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "input_rows": (ctypes.c_int, "in"),
            "input_cols": (ctypes.c_int, "in"),
            "kernel_rows": (ctypes.c_int, "in"),
            "kernel_cols": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_rows, input_cols = 5, 5
        kernel_rows, kernel_cols = 3, 3
        input = torch.tensor([
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0
        ], device="cuda", dtype=dtype)
        kernel = torch.tensor([
            0.0625, 0.125, 0.0625,
            0.125, 0.25, 0.125,
            0.0625, 0.125, 0.0625
        ], device="cuda", dtype=dtype)
        output = torch.empty(input_rows * input_cols, device="cuda", dtype=dtype)
        return {
            "input": input,
            "kernel": kernel,
            "output": output,
            "input_rows": input_rows,
            "input_cols": input_cols,
            "kernel_rows": kernel_rows,
            "kernel_cols": kernel_cols
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = "cuda"
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0]
            ], device=device, dtype=dtype).flatten(),
            "kernel": torch.tensor([
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]
            ], device=device, dtype=dtype).flatten(),
            "output": torch.zeros((5,5), device=device, dtype=dtype).flatten(),
            "input_rows": 5,
            "input_cols": 5,
            "kernel_rows": 3,
            "kernel_cols": 3
        })

        # identity_kernel
        tests.append({
            "input": torch.tensor([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ], device=device, dtype=dtype).flatten(),
            "kernel": torch.tensor([
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0]
            ], device=device, dtype=dtype).flatten(),
            "output": torch.zeros((3,3), device=device, dtype=dtype).flatten(),
            "input_rows": 3,
            "input_cols": 3,
            "kernel_rows": 3,
            "kernel_cols": 3
        })

        # all_ones_input
        tests.append({
            "input": torch.ones((4,4), device=device, dtype=dtype).flatten(),
            "kernel": torch.full((3,3), 0.111111, device=device, dtype=dtype).flatten(),
            "output": torch.zeros((4,4), device=device, dtype=dtype).flatten(),
            "input_rows": 4,
            "input_cols": 4,
            "kernel_rows": 3,
            "kernel_cols": 3
        })

        # single_pixel
        tests.append({
            "input": torch.tensor([[42.0]], device=device, dtype=dtype).flatten(),
            "kernel": torch.tensor([[1.0]], device=device, dtype=dtype).flatten(),
            "output": torch.zeros((1,1), device=device, dtype=dtype).flatten(),
            "input_rows": 1,
            "input_cols": 1,
            "kernel_rows": 1,
            "kernel_cols": 1
        })

        # large_random
        input_large = torch.empty((32,32), device=device, dtype=dtype).uniform_(-10.0,10.0)
        kernel_large = torch.empty((5,5), device=device, dtype=dtype).uniform_(0.0,1.0)
        tests.append({
            "input": input_large.flatten(),
            "kernel": kernel_large.flatten(),
            "output": torch.zeros((32,32), device=device, dtype=dtype).flatten(),
            "input_rows": 32,
            "input_cols": 32,
            "kernel_rows": 5,
            "kernel_cols": 5
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_rows, input_cols = 512, 512
        kernel_rows, kernel_cols = 7, 7
        input = torch.empty(input_rows * input_cols, device="cuda", dtype=dtype).uniform_(0.0, 255.0)
        kernel = torch.empty(kernel_rows * kernel_cols, device="cuda", dtype=dtype).uniform_(0.0001, 0.02)
        output = torch.empty(input_rows * input_cols, device="cuda", dtype=dtype)
        return {
            "input": input,
            "kernel": kernel,
            "output": output,
            "input_rows": input_rows,
            "input_cols": input_cols,
            "kernel_rows": kernel_rows,
            "kernel_cols": kernel_cols
        }
