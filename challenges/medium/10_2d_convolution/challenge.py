import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="2D Convolution",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor,
                       input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
        # Reshape flattened arrays to 2D matrices
        input_2d = input.view(input_rows, input_cols)
        kernel_2d = kernel.view(kernel_rows, kernel_cols)
        # Prepare tensors for conv2d (add batch and channel dimensions)
        kernel_prepared = kernel_2d.unsqueeze(0).unsqueeze(0)
        input_prepared = input_2d.unsqueeze(0).unsqueeze(0)
        # Perform cross-correlation using PyTorch's F.conv2d (which does cross-correlation by default)
        result = torch.nn.functional.conv2d(input_prepared, kernel_prepared, padding=0)
        # Copy result to output tensor (removing the extra dimensions and flattening)
        output.copy_(result.view(-1))

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
        input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device="cuda", dtype=dtype)
        kernel = torch.tensor([0.0, 1.0, 1.0, 0.0], device="cuda", dtype=dtype)
        output = torch.empty(4, device="cuda", dtype=dtype)
        return {
            "input": input,
            "kernel": kernel,
            "output": output,
            "input_rows": 3,
            "input_cols": 3,
            "kernel_rows": 2,
            "kernel_cols": 2
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device="cuda", dtype=dtype),
            "kernel": torch.tensor([0.0, 1.0, 1.0, 0.0], device="cuda", dtype=dtype),
            "output": torch.empty(4, device="cuda", dtype=dtype),
            "input_rows": 3,
            "input_cols": 3,
            "kernel_rows": 2,
            "kernel_cols": 2
        })
        # rectangular_input
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda", dtype=dtype),
            "kernel": torch.tensor([1.0, 0.0], device="cuda", dtype=dtype),
            "output": torch.empty(4, device="cuda", dtype=dtype),
            "input_rows": 2,
            "input_cols": 3,
            "kernel_rows": 1,
            "kernel_cols": 2
        })
        # negative_kernel
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device="cuda", dtype=dtype),
            "kernel": torch.tensor([-1.0, 1.0, 0.0, 0.0], device="cuda", dtype=dtype),
            "output": torch.empty(4, device="cuda", dtype=dtype),
            "input_rows": 3,
            "input_cols": 3,
            "kernel_rows": 2,
            "kernel_cols": 2
        })
        # single_element_kernel
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], device="cuda", dtype=dtype),
            "kernel": torch.tensor([2.0], device="cuda", dtype=dtype),
            "output": torch.empty(9, device="cuda", dtype=dtype),
            "input_rows": 3,
            "input_cols": 3,
            "kernel_rows": 1,
            "kernel_cols": 1
        })
        # medium_matrix_small_kernel
        tests.append({
            "input": torch.empty(64*64, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "kernel": torch.empty(3*3, device="cuda", dtype=dtype).uniform_(-0.5, 0.5),
            "output": torch.empty(62*62, device="cuda", dtype=dtype),
            "input_rows": 64,
            "input_cols": 64,
            "kernel_rows": 3,
            "kernel_cols": 3
        })
        # large_matrix_medium_kernel
        tests.append({
            "input": torch.empty(128*128, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
            "kernel": torch.empty(7*7, device="cuda", dtype=dtype).uniform_(-0.2, 0.2),
            "output": torch.empty(122*122, device="cuda", dtype=dtype),
            "input_rows": 128,
            "input_cols": 128,
            "kernel_rows": 7,
            "kernel_cols": 7
        })
        # rectangular_large_matrix
        tests.append({
            "input": torch.empty(128*256, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "kernel": torch.empty(5*5, device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "output": torch.empty(124*252, device="cuda", dtype=dtype),
            "input_rows": 128,
            "input_cols": 256,
            "kernel_rows": 5,
            "kernel_cols": 5
        })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_rows = 3072
        input_cols = 3072
        kernel_rows = 15
        kernel_cols = 15
        input = torch.empty(input_rows * input_cols, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        kernel = torch.empty(kernel_rows * kernel_cols, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        output_rows = input_rows - kernel_rows + 1
        output_cols = input_cols - kernel_cols + 1
        output = torch.empty(output_rows * output_cols, device="cuda", dtype=dtype)
        return {
            "input": input,
            "kernel": kernel,
            "output": output,
            "input_rows": input_rows,
            "input_cols": input_cols,
            "kernel_rows": kernel_rows,
            "kernel_cols": kernel_cols
        } 