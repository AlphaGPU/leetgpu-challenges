import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Monte Carlo Integration",
            atol=1e-02,
            rtol=1e-02,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, y_samples: torch.Tensor, result: torch.Tensor, a: float, b: float, n_samples: int):
        assert y_samples.shape == (n_samples,)
        assert result.shape == (1,)
        assert y_samples.dtype == result.dtype
        assert y_samples.device == result.device
        assert b > a
        
        # Monte Carlo integration: integral ≈ (b - a) * mean(y_samples)
        mean_y = torch.mean(y_samples)
        integral = (b - a) * mean_y
        
        result[0] = integral
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "y_samples": (ctypes.POINTER(ctypes.c_float), "in"),
            "result": (ctypes.POINTER(ctypes.c_float), "out"),
            "a": (ctypes.c_float, "in"),
            "b": (ctypes.c_float, "in"),
            "n_samples": (ctypes.c_int, "in")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        y_samples = torch.tensor([0.0625, 0.25, 0.5625, 1.0, 1.5625, 2.25, 3.0625, 4.0], device="cuda", dtype=dtype)
        result = torch.zeros(1, device="cuda", dtype=dtype)
        return {
            "y_samples": y_samples,
            "result": result,
            "a": 0.0,
            "b": 2.0,
            "n_samples": 8
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_specs = [
            # Basic test cases
            ("basic_8", [0.0625, 0.25, 0.5625, 1.0, 1.5625, 2.25, 3.0625, 4.0], 0.0, 2.0),
            ("constant_function", [1.0, 1.0, 1.0, 1.0], 0.0, 4.0),
            ("linear_function", [0.0, 1.0, 2.0, 3.0], 0.0, 3.0),
            ("negative_interval", [-1.0, -2.0, -3.0], -2.0, 1.0),
            ("small_interval", [0.5, 1.5], 1.0, 2.0),
        ]
        
        test_cases = []
        for _, y_vals, a, b in test_specs:
            n_samples = len(y_vals)
            test_cases.append({
                "y_samples": torch.tensor(y_vals, device="cuda", dtype=dtype),
                "result": torch.zeros(1, device="cuda", dtype=dtype),
                "a": a,
                "b": b,
                "n_samples": n_samples
            })
        
        # Random test cases with different sizes
        for _, n_samples, a, b in [
            ("small_samples", 10, 0.0, 1.0),
            ("medium_samples", 100, -1.0, 1.0),
            ("large_samples", 1000, 0.0, 10.0),
            ("many_samples", 10000, -5.0, 5.0),
        ]:
            test_cases.append({
                "y_samples": torch.empty(n_samples, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "result": torch.zeros(1, device="cuda", dtype=dtype),
                "a": a,
                "b": b,
                "n_samples": n_samples
            })
        
        # Edge cases
        for _, n_samples, a, b in [
            ("min_samples", 1, 0.0, 1.0),
            ("large_interval", 100, -100.0, 100.0),
            ("small_interval_edge", 50, 0.0, 0.1),
        ]:
            test_cases.append({
                "y_samples": torch.empty(n_samples, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "result": torch.zeros(1, device="cuda", dtype=dtype),
                "a": a,
                "b": b,
                "n_samples": n_samples
            })
        
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        n_samples = 10000000  
        return {
            "y_samples": torch.empty(n_samples, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "result": torch.zeros(1, device="cuda", dtype=dtype),
            "a": -10.0,
            "b": 10.0,
            "n_samples": n_samples
        } 