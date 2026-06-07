import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandTensor


class Challenge(ChallengeBase):
    name = "3D Convolution"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        input: torch.Tensor,
        kernel: torch.Tensor,
        output: torch.Tensor,
        input_depth: int,
        input_rows: int,
        input_cols: int,
        kernel_depth: int,
        kernel_rows: int,
        kernel_cols: int,
    ):
        assert input.shape == (input_depth, input_rows, input_cols)
        assert kernel.shape == (kernel_depth, kernel_rows, kernel_cols)
        assert output.shape == (
            input_depth - kernel_depth + 1,
            input_rows - kernel_rows + 1,
            input_cols - kernel_cols + 1,
        )
        assert input.dtype == kernel.dtype == output.dtype
        assert input.device == kernel.device == output.device

        input_expanded = input.unsqueeze(0).unsqueeze(0)
        kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)

        result = torch.nn.functional.conv3d(
            input_expanded, kernel_expanded, bias=None, stride=1, padding=0
        )

        output.copy_(result.squeeze(0).squeeze(0))

    def reference_impl_jax(
        self,
        input,
        kernel,
        input_depth,
        input_rows,
        input_cols,
        kernel_depth,
        kernel_rows,
        kernel_cols,
    ):
        import jax

        # Cross-correlation (matches F.conv3d), valid padding, stride 1.
        # Shapes: input (N=1, C=1, D, H, W), kernel (O=1, I=1, D, H, W).
        lhs = input.reshape(1, 1, input_depth, input_rows, input_cols)
        rhs = kernel.reshape(1, 1, kernel_depth, kernel_rows, kernel_cols)
        result = jax.lax.conv_general_dilated(
            lhs,
            rhs,
            window_strides=(1, 1, 1),
            padding="VALID",
            precision=jax.lax.Precision.HIGHEST,
        )
        return result.reshape(result.shape[2:])

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "kernel": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "input_depth": (ctypes.c_int, "in"),
            "input_rows": (ctypes.c_int, "in"),
            "input_cols": (ctypes.c_int, "in"),
            "kernel_depth": (ctypes.c_int, "in"),
            "kernel_rows": (ctypes.c_int, "in"),
            "kernel_cols": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_tensor = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ],
            dtype=dtype,
            device=self.device,
        )
        kernel_tensor = torch.tensor(
            [[[1, 0, 0], [1, 1, 1], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 1]]],
            dtype=dtype,
            device=self.device,
        )
        output_tensor = torch.empty((2, 1, 1), device=self.device, dtype=dtype)
        return {
            "input": input_tensor,
            "kernel": kernel_tensor,
            "output": output_tensor,
            "input_depth": 3,
            "input_rows": 3,
            "input_cols": 3,
            "kernel_depth": 2,
            "kernel_rows": 3,
            "kernel_cols": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = self.device
        tests = []

        # basic_example
        tests.append(
            {
                "input": torch.tensor(
                    [
                        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
                    ],
                    dtype=dtype,
                    device=device,
                ),
                "kernel": torch.tensor(
                    [[[1, 0, 0], [1, 1, 1], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 1]]],
                    dtype=dtype,
                    device=device,
                ),
                "output": torch.zeros((2, 1, 1), dtype=dtype, device=device),
                "input_depth": 3,
                "input_rows": 3,
                "input_cols": 3,
                "kernel_depth": 2,
                "kernel_rows": 3,
                "kernel_cols": 3,
            }
        )

        # small_dimensions
        tests.append(
            {
                "input": torch.tensor(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype, device=device
                ),
                "kernel": torch.tensor(
                    [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=dtype, device=device
                ),
                "output": torch.zeros((1, 1, 1), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 2,
                "input_cols": 2,
                "kernel_depth": 2,
                "kernel_rows": 2,
                "kernel_cols": 2,
            }
        )

        # unit_kernel
        tests.append(
            {
                "input": torch.tensor(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype, device=device
                ),
                "kernel": torch.tensor([[[2]]], dtype=dtype, device=device),
                "output": torch.zeros((2, 2, 2), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 2,
                "input_cols": 2,
                "kernel_depth": 1,
                "kernel_rows": 1,
                "kernel_cols": 1,
            }
        )

        # zero_kernel
        tests.append(
            {
                "input": torch.tensor(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype, device=device
                ),
                "kernel": torch.zeros((2, 2, 2), dtype=dtype, device=device),
                "output": torch.zeros((1, 1, 1), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 2,
                "input_cols": 2,
                "kernel_depth": 2,
                "kernel_rows": 2,
                "kernel_cols": 2,
            }
        )

        # negative_values
        tests.append(
            {
                "input": torch.tensor(
                    [[[-1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=dtype, device=device
                ),
                "kernel": torch.tensor([[[-1, 1], [-1, 1]]], dtype=dtype, device=device),
                "output": torch.zeros((2, 1, 1), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 2,
                "input_cols": 2,
                "kernel_depth": 1,
                "kernel_rows": 2,
                "kernel_cols": 2,
            }
        )

        # rectangular_dimensions
        tests.append(
            {
                "input": torch.tensor(
                    [
                        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
                    ],
                    dtype=dtype,
                    device=device,
                ),
                "kernel": torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=dtype, device=device),
                "output": torch.zeros((2, 2, 2), dtype=dtype, device=device),
                "input_depth": 2,
                "input_rows": 3,
                "input_cols": 4,
                "kernel_depth": 1,
                "kernel_rows": 2,
                "kernel_cols": 3,
            }
        )

        # power_of_two_dimensions
        tests.append(
            {
                "input": torch.empty(4, 4, 4, device=device, dtype=dtype).uniform_(-1.0, 1.0),
                "kernel": torch.empty(3, 3, 3, device=device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros((2, 2, 2), dtype=dtype, device=device),
                "input_depth": 4,
                "input_rows": 4,
                "input_cols": 4,
                "kernel_depth": 3,
                "kernel_rows": 3,
                "kernel_cols": 3,
            }
        )

        # medium_size
        tests.append(
            {
                "input": torch.empty(10, 10, 10, device=device, dtype=dtype).uniform_(-10.0, 10.0),
                "kernel": torch.empty(3, 4, 5, device=device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros((8, 7, 6), dtype=dtype, device=device),
                "input_depth": 10,
                "input_rows": 10,
                "input_cols": 10,
                "kernel_depth": 3,
                "kernel_rows": 4,
                "kernel_cols": 5,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        input_depth, input_rows, input_cols = 256, 128, 128
        kernel_depth, kernel_rows, kernel_cols = 5, 5, 5
        return {
            "input": RandTensor((input_depth, input_rows, input_cols), -1.0, 1.0),
            "kernel": RandTensor((kernel_depth, kernel_rows, kernel_cols), -1.0, 1.0),
            "output": OutTensor(
                (
                    input_depth - kernel_depth + 1,
                    input_rows - kernel_rows + 1,
                    input_cols - kernel_cols + 1,
                )
            ),
            "input_depth": input_depth,
            "input_rows": input_rows,
            "input_cols": input_cols,
            "kernel_depth": kernel_depth,
            "kernel_rows": kernel_rows,
            "kernel_cols": kernel_cols,
        }
