import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Matrix Power",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, P: int):
        """
        Matrix power implementation using PyTorch.
        Raises an N x N matrix to integer power P.
        """
        assert input.dtype == torch.float32
        assert output.dtype == torch.float32
        assert input.shape == output.shape == (N * N,)
        assert P >= 1

        mat = input.view(N, N)
        result = torch.linalg.matrix_power(mat, P).float()
        output[:] = result.reshape(-1)


    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "P": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 2
        P = 3
        input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype).flatten()
        output_data = torch.zeros((2,2), device="cuda", dtype=dtype).flatten()

        return {
            "input": input_data,
            "output": output_data,
            "N": N,
            "P": P
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []

        # Test case 1: example 2x2 power 3
        test_cases.append({
            "input": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype).flatten(),
            "output": torch.zeros((2, 2), device="cuda", dtype=dtype).flatten(),
            "N": 2,
            "P": 3
        })

        # Test case 2: identity 3x3 power 5
        test_cases.append({
            "input": torch.eye(3, device="cuda", dtype=dtype).flatten(),
            "output": torch.zeros((3, 3), device="cuda", dtype=dtype).flatten(),
            "N": 3,
            "P": 5
        })

        # Test case 3: random 5x5 power 2
        test_cases.append({
            "input": torch.empty((5, 5), device="cuda", dtype=dtype).uniform_(-5.0, 5.0).flatten(),
            "output": torch.zeros((5, 5), device="cuda", dtype=dtype).flatten(),
            "N": 5,
            "P": 2
        })

        # Test case 4: random 16x16 power 3
        test_cases.append({
            "input": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0).flatten(),
            "output": torch.zeros((16, 16), device="cuda", dtype=dtype).flatten(),
            "N": 16,
            "P": 3
        })

        # Test case 5: random 8x8 power 4
        test_cases.append({
            "input": torch.empty((8, 8), device="cuda", dtype=dtype).uniform_(-10.0, 10.0).flatten(),
            "output": torch.zeros((8, 8), device="cuda", dtype=dtype).flatten(),
            "N": 8,
            "P": 4
        })

        # Test case 6: random 10x10 power 1
        test_cases.append({
            "input": torch.empty((10, 10), device="cuda", dtype=dtype).uniform_(-2.0, 2.0).flatten(),
            "output": torch.zeros((10, 10), device="cuda", dtype=dtype).flatten(),
            "N": 10,
            "P": 1
        })

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 512
        P = 3
        return {
            "input": torch.empty((N, N), device="cuda", dtype=dtype).uniform_(-10.0, 10.0).flatten(),
            "output": torch.zeros((N, N), device="cuda", dtype=dtype).flatten(),
            "N": N,
            "P": P
        }
