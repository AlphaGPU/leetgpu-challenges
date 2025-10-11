import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Vector Addition",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
        assert A.shape == B.shape == C.shape
        assert A.dtype == B.dtype == C.dtype
        assert A.device == B.device == C.device
        
        torch.add(A, B, out=C)
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_size_t, "in")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        A = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        B = torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype)
        C = torch.empty(N, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "N": N
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_specs = [
            ("basic_small", [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]),
            ("all_zeros", [0.0] * 16, [0.0] * 16),
            ("non_power_of_two", [1.0] * 30, [2.0] * 30),
            ("negative_numbers", [-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]),
            ("mixed_positive_negative", [1.0, -2.0, 3.0, -4.0], [-1.0, 2.0, -3.0, 4.0]),
            ("very_small_numbers", [0.000001, 0.0000001, 0.00000001, 0.000000001], [0.000001, 0.0000001, 0.00000001, 0.000000001]),
            ("large_numbers", [1000000.0, 10000000.0, -1000000.0, -10000000.0], [1000000.0, -10000000.0, -1000000.0, 10000000.0]),
        ]
        
        test_cases = []
        for _, a_vals, b_vals in test_specs:
            n = len(a_vals)
            test_cases.append({
                "A": torch.tensor(a_vals, device="cuda", dtype=dtype),
                "B": torch.tensor(b_vals, device="cuda", dtype=dtype),
                "C": torch.zeros(n, device="cuda", dtype=dtype),
                "N": n
            })
        
        # Random test cases
        for _, size, a_range, b_range in [
            ("powers_of_two_size", 32, (0.0, 32.0), (0.0, 64.0)),
            ("medium_sized_vector", 1000, (0.0, 7.0), (0.0, 5.0)),
            ("large_vector", 10000, (0.0, 1.0), (0.0, 1.0)),
        ]:
            test_cases.append({
                "A": torch.empty(size, device="cuda", dtype=dtype).uniform_(*a_range),
                "B": torch.empty(size, device="cuda", dtype=dtype).uniform_(*b_range),
                "C": torch.zeros(size, device="cuda", dtype=dtype),
                "N": size
            })
        
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 25000000
        return {
            "A": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "B": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "C": torch.zeros(N, device="cuda", dtype=dtype),
            "N": N
        }
    
