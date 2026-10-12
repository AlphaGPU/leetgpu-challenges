import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "GELU Activation"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, X: torch.Tensor, Y: torch.Tensor, N: int):
        assert X.shape == Y.shape
        assert X.shape == (N,)
        assert X.dtype == torch.float32
        assert Y.dtype == torch.float32

        Y.copy_(torch.nn.functional.gelu(X, approximate="none"))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "Y": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_size_t, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        X = torch.tensor([0.0, 1.0, -1.0, 2.0], device=self.device, dtype=dtype)
        Y = torch.empty(N, device=self.device, dtype=dtype)
        return {
            "X": X,
            "Y": Y,
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32

        test_specs = [
            ("single_zero", [0.0]),
            ("single_negative", [-3.0]),
            ("basic_small", [0.0, 1.0, -1.0, 2.0]),
            ("all_zeros", [0.0] * 16),
            ("large_magnitudes", [10.0, -10.0, 50.0, -50.0]),
            ("mixed_values", [0.5, -0.5, 1.5, -1.5, 3.0, -3.0, 0.0, 7.0]),
        ]

        test_cases = []
        for _, x_vals in test_specs:
            n = len(x_vals)
            test_cases.append(
                {
                    "X": torch.tensor(x_vals, device=self.device, dtype=dtype),
                    "Y": torch.empty(n, device=self.device, dtype=dtype),
                    "N": n,
                }
            )

        for size, low, high in [
            (64, -5.0, 5.0),
            (255, -10.0, 10.0),
            (10000, -4.0, 4.0),
        ]:
            test_cases.append(
                {
                    "X": torch.empty(size, device=self.device, dtype=dtype).uniform_(low, high),
                    "Y": torch.empty(size, device=self.device, dtype=dtype),
                    "N": size,
                }
            )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000000
        return {
            "X": torch.empty(N, device=self.device, dtype=dtype).uniform_(-10.0, 10.0),
            "Y": torch.empty(N, device=self.device, dtype=dtype),
            "N": N,
        }
