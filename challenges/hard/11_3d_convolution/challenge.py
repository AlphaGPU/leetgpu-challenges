import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="3D Convolution",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor,
                       input_depth: int, input_rows: int, input_cols: int,
                       kernel_depth: int, kernel_rows: int, kernel_cols: int):
        assert input.shape == (input_depth, input_rows, input_cols)
        assert kernel.shape == (kernel_depth, kernel_rows, kernel_cols)
        assert output.shape == (
            input_depth - kernel_depth + 1,
            input_rows - kernel_rows + 1,
            input_cols - kernel_cols + 1
        )
        assert input.dtype == kernel.dtype == output.dtype
        assert input.device == kernel.device == output.device
        
        input_expanded = input.unsqueeze(0).unsqueeze(0)
        kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)
        
        result = torch.nn.functional.conv3d(
            input_expanded, 
            kernel_expanded, 
            bias=None, 
            stride=1, 
            padding=0
        )
        
        output.copy_(result.squeeze(0).squeeze(0))

    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "input": ctypes.POINTER(ctypes.c_float),
            "kernel": ctypes.POINTER(ctypes.c_float),
            "output": ctypes.POINTER(ctypes.c_float),
            "input_depth": ctypes.c_int,
            "input_rows": ctypes.c_int,
            "input_cols": ctypes.c_int,
            "kernel_depth": ctypes.c_int,
            "kernel_rows": ctypes.c_int,
            "kernel_cols": ctypes.c_int
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_tensor = torch.tensor([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
        ], dtype=dtype, device="cuda")
        kernel_tensor = torch.tensor([
          [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
          [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
        ], dtype=dtype, device="cuda")
        output_tensor = torch.empty((2, 1, 1), device="cuda", dtype=dtype)
        return {
            "input": input_tensor,
            "kernel": kernel_tensor,
            "output": output_tensor,
            "input_depth": 3,
            "input_rows": 3,
            "input_cols": 3,
            "kernel_depth": 2,
            "kernel_rows": 3,
            "kernel_cols": 3
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = "cuda"
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([
                [[1,2,3],[4,5,6],[7,8,9]],
                [[10,11,12],[13,14,15],[16,17,18]],
                [[19,20,21],[22,23,24],[25,26,27]]
            ], dtype=dtype, device=device),
            "kernel": torch.tensor([
                [[1,0,0],[1,1,1],[0,0,0]],
                [[1,1,0],[1,1,0],[0,0,1]]
            ], dtype=dtype, device=device),
            "output": torch.zeros((2,1,1), dtype=dtype, device=device),
            "input_depth": 3,
            "input_rows": 3,
            "input_cols": 3,
            "kernel_depth": 2,
            "kernel_rows": 3,
            "kernel_cols": 3
        })

        # small_dimensions
        tests.append({
            "input": torch.tensor([
                [[1,2],[3,4]],
                [[5,6],[7,8]]
            ], dtype=dtype, device=device),
            "kernel": torch.tensor([
                [[1,1],[1,1]],
                [[1,1],[1,1]]
            ], dtype=dtype, device=device),
            "output": torch.zeros((1,1,1), dtype=dtype, device=device),
            "input_depth": 2,
            "input_rows": 2,
            "input_cols": 2,
            "kernel_depth": 2,
            "kernel_rows": 2,
            "kernel_cols": 2
        })

        # unit_kernel
        tests.append({
            "input": torch.tensor([
                [[1,2],[3,4]],
                [[5,6],[7,8]]
            ], dtype=dtype, device=device),
            "kernel": torch.tensor([[[2]]], dtype=dtype, device=device),
            "output": torch.zeros((2,2,2), dtype=dtype, device=device),
            "input_depth": 2,
            "input_rows": 2,
            "input_cols": 2,
            "kernel_depth": 1,
            "kernel_rows": 1,
            "kernel_cols": 1
        })

        # zero_kernel
        tests.append({
            "input": torch.tensor([
                [[1,2],[3,4]],
                [[5,6],[7,8]]
            ], dtype=dtype, device=device),
            "kernel": torch.zeros((2,2,2), dtype=dtype, device=device),
            "output": torch.zeros((1,1,1), dtype=dtype, device=device),
            "input_depth": 2,
            "input_rows": 2,
            "input_cols": 2,
            "kernel_depth": 2,
            "kernel_rows": 2,
            "kernel_cols": 2
        })

        # negative_values
        tests.append({
            "input": torch.tensor([
                [[-1,-2],[3,-4]],
                [[5,-6],[7,-8]]
            ], dtype=dtype, device=device),
            "kernel": torch.tensor([
                [[-1,1],[-1,1]]
            ], dtype=dtype, device=device),
            "output": torch.zeros((2,1,1), dtype=dtype, device=device),
            "input_depth": 2,
            "input_rows": 2,
            "input_cols": 2,
            "kernel_depth": 1,
            "kernel_rows": 2,
            "kernel_cols": 2
        })

        # rectangular_dimensions
        tests.append({
            "input": torch.tensor([
                [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                [[13,14,15,16],[17,18,19,20],[21,22,23,24]]
            ], dtype=dtype, device=device),
            "kernel": torch.tensor([
                [[1,1,1],[1,1,1]]
            ], dtype=dtype, device=device),
            "output": torch.zeros((2,2,2), dtype=dtype, device=device),
            "input_depth": 2,
            "input_rows": 3,
            "input_cols": 4,
            "kernel_depth": 1,
            "kernel_rows": 2,
            "kernel_cols": 3
        })

        # power_of_two_dimensions
        tests.append({
            "input": torch.empty(4,4,4, device=device, dtype=dtype).uniform_(-1.0,1.0),
            "kernel": torch.empty(3,3,3, device=device, dtype=dtype).uniform_(-1.0,1.0),
            "output": torch.zeros((2,2,2), dtype=dtype, device=device),
            "input_depth": 4,
            "input_rows": 4,
            "input_cols": 4,
            "kernel_depth": 3,
            "kernel_rows": 3,
            "kernel_cols": 3
        })

        # medium_size
        tests.append({
            "input": torch.empty(10,10,10, device=device, dtype=dtype).uniform_(-10.0,10.0),
            "kernel": torch.empty(3,4,5, device=device, dtype=dtype).uniform_(-1.0,1.0),
            "output": torch.zeros((8,7,6), dtype=dtype, device=device),
            "input_depth": 10,
            "input_rows": 10,
            "input_cols": 10,
            "kernel_depth": 3,
            "kernel_rows": 4,
            "kernel_cols": 5
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_depth, input_rows, input_cols = 256, 128, 128
        kernel_depth, kernel_rows, kernel_cols = 5, 5, 5
        input_tensor = torch.empty(input_depth, input_rows, input_cols, device="cuda", dtype=dtype).uniform_(-1.0,1.0)
        kernel_tensor = torch.empty(kernel_depth, kernel_rows, kernel_cols, device="cuda", dtype=dtype).uniform_(-1.0,1.0)
        output_tensor = torch.zeros(
            input_depth - kernel_depth + 1,
            input_rows - kernel_rows + 1,
            input_cols - kernel_cols + 1,
            device="cuda", dtype=dtype
        )
        return {
            "input": input_tensor,
            "kernel": kernel_tensor,
            "output": output_tensor,
            "input_depth": input_depth,
            "input_rows": input_rows,
            "input_cols": input_cols,
            "kernel_depth": kernel_depth,
            "kernel_rows": kernel_rows,
            "kernel_cols": kernel_cols
        }
