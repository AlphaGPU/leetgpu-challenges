import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Matrix Addition",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
        assert A.shape == (N, N)
        assert B.shape == (N, N)
        assert C.shape == (N, N)
        assert A.dtype == B.dtype == C.dtype
        assert A.device == B.device == C.device
        
        torch.add(A, B, out=C)
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 2
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype)
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype)
        C = torch.empty(N, N, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "N": N
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []
        
        # basic_2x2
        test_cases.append({
            "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2
        })
        
        # all_zeros_4x4
        test_cases.append({
            "A": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "B": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "C": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "N": 4
        })
        
        # identity_plus_identity_3x3
        test_cases.append({
            "A": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((3, 3), device="cuda", dtype=dtype),
            "N": 3
        })
        
        # negative_values_2x2
        test_cases.append({
            "A": torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[-5.0, -6.0], [-7.0, -8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2
        })
        
        # mixed_positive_negative_2x2
        test_cases.append({
            "A": torch.tensor([[1.0, -2.0], [-3.0, 4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[-1.0, 2.0], [3.0, -4.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2
        })
        
        # single_element_1x1
        test_cases.append({
            "A": torch.tensor([[42.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((1, 1), device="cuda", dtype=dtype),
            "N": 1
        })
        
        # large_N_16x16
        test_cases.append({
            "A": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "C": torch.zeros((16, 16), device="cuda", dtype=dtype),
            "N": 16
        })
        
        # very_small_numbers
        test_cases.append({
            "A": torch.tensor([[0.000001, 0.0000001], [0.00000001, 0.000000001]], device="cuda", dtype=dtype),
            "B": torch.tensor([[0.000001, 0.0000001], [0.00000001, 0.000000001]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2
        })
        
        # large_numbers
        test_cases.append({
            "A": torch.tensor([[1000000.0, 10000000.0], [-1000000.0, -10000000.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[1000000.0, -10000000.0], [-1000000.0, 10000000.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2
        })
        
        # non_power_of_two_size_7x7
        test_cases.append({
            "A": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
            "B": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
            "C": torch.zeros((7, 7), device="cuda", dtype=dtype),
            "N": 7
        })
        
        # medium_size_32x32
        test_cases.append({
            "A": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "B": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "C": torch.zeros((32, 32), device="cuda", dtype=dtype),
            "N": 32
        })
        
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4096
        return {
            "A": torch.empty(N, N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "B": torch.empty(N, N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "C": torch.zeros(N, N, device="cuda", dtype=dtype),
            "N": N
        }
    
