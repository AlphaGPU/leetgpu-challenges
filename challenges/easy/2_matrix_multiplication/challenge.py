import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Matrix Multiplication",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
        assert A.shape == (M, N)
        assert B.shape == (N, K)
        assert C.shape == (M, K)
        assert A.dtype == B.dtype == C.dtype
        assert A.device == B.device == C.device
        
        torch.matmul(A, B, out=C)
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N, K = 2, 2, 2
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype)
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype)
        C = torch.empty(M, K, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "M": M,
            "N": N,
            "K": K
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_specs = [
            # Basic test cases
            ("basic_2x2", 2, 2, 2, [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
            ("basic_1x3_3x1", 1, 3, 1, [[1.0, 2.0, 3.0]], [[4.0], [5.0], [6.0]]),
            ("identity_matrix", 3, 3, 3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 
             [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            ("zero_matrix", 2, 2, 2, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]),
            ("rectangular_matrices", 2, 3, 1, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0], [2.0], [3.0]]),
        ]
        
        test_cases = []
        for _, m, n, k, a_vals, b_vals in test_specs:
            test_cases.append({
                "A": torch.tensor(a_vals, device="cuda", dtype=dtype),
                "B": torch.tensor(b_vals, device="cuda", dtype=dtype),
                "C": torch.empty(m, k, device="cuda", dtype=dtype),
                "M": m,
                "N": n,
                "K": k
            })
        
        # Random test cases with different sizes
        for _, m, n, k in [
            ("small_square", 4, 4, 4),
            ("medium_rectangular", 8, 6, 10),
            ("large_rectangular", 16, 12, 20),
            ("tall_matrix", 32, 8, 16),
            ("wide_matrix", 8, 16, 32),
        ]:
            test_cases.append({
                "A": torch.empty(m, n, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "B": torch.empty(n, k, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "C": torch.empty(m, k, device="cuda", dtype=dtype),
                "M": m,
                "N": n,
                "K": k
            })
        
        # Edge cases
        for _, m, n, k in [
            ("single_element", 1, 1, 1),
            ("single_row", 1, 5, 3),
            ("single_column", 5, 3, 1),
            ("max_dimensions", 8192, 6144, 4096),
        ]:
            test_cases.append({
                "A": torch.empty(m, n, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "B": torch.empty(n, k, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty(m, k, device="cuda", dtype=dtype),
                "M": m,
                "N": n,
                "K": k
            })
        
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N, K = 8192, 6144, 4096 
        return {
            "A": torch.empty(M, N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.empty(N, K, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "C": torch.empty(M, K, device="cuda", dtype=dtype),
            "M": M,
            "N": N,
            "K": K
        } 