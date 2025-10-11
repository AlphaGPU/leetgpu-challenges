import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Mean Squared Error",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
        # predictions, targets, mse are tensors on the GPU
        squared_diffs = torch.square(predictions - targets)
        mean_squared_error = torch.mean(squared_diffs)
        mse[0] = mean_squared_error

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "predictions": (ctypes.POINTER(ctypes.c_float), "in"),
            "targets": (ctypes.POINTER(ctypes.c_float), "in"),
            "mse": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        targets = torch.tensor([1.5, 2.5, 3.5, 4.5], device="cuda", dtype=dtype)
        mse = torch.empty(1, device="cuda", dtype=dtype)
        N = 4
        return {
            "predictions": predictions,
            "targets": targets,
            "mse": mse,
            "N": N
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # Test 1: basic_example
        tests.append({
            "predictions": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype),
            "targets": torch.tensor([1.5, 2.5, 3.5, 4.5], device="cuda", dtype=dtype),
            "mse": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4
        })
        # Test 2: second_example
        tests.append({
            "predictions": torch.tensor([10.0, 20.0, 30.0], device="cuda", dtype=dtype),
            "targets": torch.tensor([12.0, 18.0, 33.0], device="cuda", dtype=dtype),
            "mse": torch.empty(1, device="cuda", dtype=dtype),
            "N": 3
        })
        # Test 3: zero_error
        tests.append({
            "predictions": torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], device="cuda", dtype=dtype),
            "targets": torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], device="cuda", dtype=dtype),
            "mse": torch.empty(1, device="cuda", dtype=dtype),
            "N": 5
        })
        # Test 4: negative_values
        tests.append({
            "predictions": torch.tensor([-2.5, -1.0, 0.0, 1.5], device="cuda", dtype=dtype),
            "targets": torch.tensor([-1.5, -2.0, 0.5, 2.0], device="cuda", dtype=dtype),
            "mse": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4
        })
        # Test 5: large_difference
        tests.append({
            "predictions": torch.tensor([100.0, 200.0, 300.0], device="cuda", dtype=dtype),
            "targets": torch.tensor([150.0, 250.0, 350.0], device="cuda", dtype=dtype),
            "mse": torch.empty(1, device="cuda", dtype=dtype),
            "N": 3
        })
        # Test 6: medium_size
        N = 1024
        predictions = torch.empty(N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0)
        targets = torch.empty(N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0)
        mse = torch.empty(1, device="cuda", dtype=dtype)
        tests.append({
            "predictions": predictions,
            "targets": targets,
            "mse": mse,
            "N": N
        })
        # Test 7: large_size
        N = 10000
        predictions = torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0)
        targets = torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0)
        mse = torch.empty(1, device="cuda", dtype=dtype)
        tests.append({
            "predictions": predictions,
            "targets": targets,
            "mse": mse,
            "N": N
        })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000000
        predictions = torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0)
        targets = torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0)
        mse = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "predictions": predictions,
            "targets": targets,
            "mse": mse,
            "N": N
        } 