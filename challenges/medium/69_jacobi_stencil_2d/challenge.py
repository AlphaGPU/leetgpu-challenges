import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="2D Jacobi Stencil",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        rows: int,
        cols: int,
    ):
        assert input.shape == (rows, cols)
        assert output.shape == (rows, cols)
        assert input.dtype == torch.float32
        assert input.device.type == "cuda"

        # Copy boundary cells unchanged
        output.copy_(input)

        # Apply 5-point stencil to interior cells:
        # output[i, j] = 0.25 * (input[i-1,j] + input[i+1,j] + input[i,j-1] + input[i,j+1])
        output[1:-1, 1:-1] = 0.25 * (
            input[0:-2, 1:-1]  # top neighbor
            + input[2:, 1:-1]  # bottom neighbor
            + input[1:-1, 0:-2]  # left neighbor
            + input[1:-1, 2:]  # right neighbor
        )

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "rows": (ctypes.c_int, "in"),
            "cols": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            device="cuda",
            dtype=dtype,
        )
        output = torch.empty((4, 4), device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "rows": 4,
            "cols": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # minimal_3x3 (only one interior cell)
        tests.append(
            {
                "input": torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    device="cuda",
                    dtype=dtype,
                ),
                "output": torch.empty((3, 3), device="cuda", dtype=dtype),
                "rows": 3,
                "cols": 3,
            }
        )

        # minimal_1x1 (all boundary, no interior cells)
        tests.append(
            {
                "input": torch.tensor([[42.0]], device="cuda", dtype=dtype),
                "output": torch.empty((1, 1), device="cuda", dtype=dtype),
                "rows": 1,
                "cols": 1,
            }
        )

        # single_row (all boundary)
        tests.append(
            {
                "input": torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda", dtype=dtype),
                "output": torch.empty((1, 4), device="cuda", dtype=dtype),
                "rows": 1,
                "cols": 4,
            }
        )

        # single_col (all boundary)
        tests.append(
            {
                "input": torch.tensor([[1.0], [2.0], [3.0], [4.0]], device="cuda", dtype=dtype),
                "output": torch.empty((4, 1), device="cuda", dtype=dtype),
                "rows": 4,
                "cols": 1,
            }
        )

        # all_zeros (interior should stay zero)
        tests.append(
            {
                "input": torch.zeros((16, 16), device="cuda", dtype=dtype),
                "output": torch.empty((16, 16), device="cuda", dtype=dtype),
                "rows": 16,
                "cols": 16,
            }
        )

        # uniform_constant (interior stays the same when all values equal)
        tests.append(
            {
                "input": torch.full((32, 32), 3.14, device="cuda", dtype=dtype),
                "output": torch.empty((32, 32), device="cuda", dtype=dtype),
                "rows": 32,
                "cols": 32,
            }
        )

        # power_of_2_square_64
        tests.append(
            {
                "input": torch.empty((64, 64), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty((64, 64), device="cuda", dtype=dtype),
                "rows": 64,
                "cols": 64,
            }
        )

        # power_of_2_square_128
        tests.append(
            {
                "input": torch.empty((128, 128), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "output": torch.empty((128, 128), device="cuda", dtype=dtype),
                "rows": 128,
                "cols": 128,
            }
        )

        # non_power_of_2_30x30
        tests.append(
            {
                "input": torch.empty((30, 30), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty((30, 30), device="cuda", dtype=dtype),
                "rows": 30,
                "cols": 30,
            }
        )

        # non_power_of_2_100x100
        tests.append(
            {
                "input": torch.empty((100, 100), device="cuda", dtype=dtype).uniform_(-3.0, 3.0),
                "output": torch.empty((100, 100), device="cuda", dtype=dtype),
                "rows": 100,
                "cols": 100,
            }
        )

        # non_square_255x33
        tests.append(
            {
                "input": torch.empty((255, 33), device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
                "output": torch.empty((255, 33), device="cuda", dtype=dtype),
                "rows": 255,
                "cols": 33,
            }
        )

        # negative_values_non_square_17x97
        tests.append(
            {
                "input": torch.empty((17, 97), device="cuda", dtype=dtype).uniform_(-100.0, 0.0),
                "output": torch.empty((17, 97), device="cuda", dtype=dtype),
                "rows": 17,
                "cols": 97,
            }
        )

        # realistic_medium_512x256
        tests.append(
            {
                "input": torch.empty((512, 256), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty((512, 256), device="cuda", dtype=dtype),
                "rows": 512,
                "cols": 256,
            }
        )

        # realistic_large_1024x1024
        tests.append(
            {
                "input": torch.empty((1024, 1024), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty((1024, 1024), device="cuda", dtype=dtype),
                "rows": 1024,
                "cols": 1024,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        rows = 8192
        cols = 8192
        return {
            "input": torch.empty((rows, cols), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "output": torch.empty((rows, cols), device="cuda", dtype=dtype),
            "rows": rows,
            "cols": cols,
        }
