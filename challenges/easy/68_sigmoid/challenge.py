import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Sigmoid Activation", atol=1e-05, rtol=1e-05, num_gpus=1, access_tier="free"
        )

    def reference_impl(self, X: torch.Tensor, Y: torch.Tensor, N: int):
        assert X.shape == Y.shape
        assert X.dtype == torch.float32
        assert Y.dtype == torch.float32
        assert X.device.type == "cuda"
        assert Y.device.type == "cuda"

        torch.sigmoid(X, out=Y)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "Y": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_size_t, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        X = torch.tensor([0.0, 1.0, -1.0, 2.0], device="cuda", dtype=dtype)
        Y = torch.empty(N, device="cuda", dtype=dtype)
        return {
            "X": X,
            "Y": Y,
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32

        test_specs = [
            ("single_zero", [0.0]),
            ("single_positive", [1.0]),
            ("single_negative", [-1.0]),
            ("basic_small", [0.0, 1.0, -1.0, 2.0]),
            ("all_zeros", [0.0] * 16),
            ("large_positives", [10.0, 20.0, 100.0, 1000.0]),
            ("large_negatives", [-10.0, -20.0, -100.0, -1000.0]),
            ("mixed_values", [0.5, -0.5, 1.5, -1.5, 3.0, -3.0, 0.0, 7.0]),
        ]

        test_cases = []
        for _, x_vals in test_specs:
            n = len(x_vals)
            test_cases.append(
                {
                    "X": torch.tensor(x_vals, device="cuda", dtype=dtype),
                    "Y": torch.empty(n, device="cuda", dtype=dtype),
                    "N": n,
                }
            )

        # Random and structured test cases
        for size, low, high in [
            (32, -5.0, 5.0),
            (100, -3.0, 3.0),
            (255, -10.0, 10.0),
            (1024, -1.0, 1.0),
            (10000, -5.0, 5.0),
        ]:
            test_cases.append(
                {
                    "X": torch.empty(size, device="cuda", dtype=dtype).uniform_(low, high),
                    "Y": torch.empty(size, device="cuda", dtype=dtype),
                    "N": size,
                }
            )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000000
        return {
            "X": torch.empty(N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "Y": torch.empty(N, device="cuda", dtype=dtype),
            "N": N,
        }
