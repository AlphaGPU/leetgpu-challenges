import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Matrix Transpose",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
        assert input.shape == (rows, cols)
        assert output.shape == (cols, rows)
        assert input.dtype == output.dtype
        assert input.device == output.device

        output.copy_(input.transpose(0, 1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "rows": (ctypes.c_int, "in"),
            "cols": (ctypes.c_int, "in")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        rows, cols = 2, 3
        input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda", dtype=dtype)
        output_tensor = torch.empty(cols, rows, device="cuda", dtype=dtype)
        return {
            "input": input_tensor,
            "output": output_tensor,
            "rows": rows,
            "cols": cols
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_specs = [
            # Basic test cases
            ("basic_2x3", 2, 3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            ("basic_3x1", 3, 1, [[1.0], [2.0], [3.0]]),
            ("square_2x2", 2, 2, [[1.0, 2.0], [3.0, 4.0]]),
            ("single_row", 1, 4, [[1.0, 2.0, 3.0, 4.0]]),
            ("single_column", 4, 1, [[1.0], [2.0], [3.0], [4.0]]),
        ]
        
        test_cases = []
        for _, r, c, input_vals in test_specs:
            test_cases.append({
                "input": torch.tensor(input_vals, device="cuda", dtype=dtype),
                "output": torch.empty(c, r, device="cuda", dtype=dtype),
                "rows": r,
                "cols": c
            })
        
        # Random test cases with different sizes
        for _, rows, cols in [
            ("small_rectangular", 4, 6),
            ("medium_square", 8, 8),
            ("large_rectangular", 16, 12),
            ("tall_matrix", 32, 8),
            ("wide_matrix", 8, 32),
        ]:
            test_cases.append({
                "input": torch.empty(rows, cols, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "output": torch.empty(cols, rows, device="cuda", dtype=dtype),
                "rows": rows,
                "cols": cols
            })
        
        # Edge cases
        for _, rows, cols in [
            ("single_element", 1, 1),
            ("max_dimensions", 8192, 8192),
        ]:
            test_cases.append({
                "input": torch.empty(rows, cols, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(cols, rows, device="cuda", dtype=dtype),
                "rows": rows,
                "cols": cols
            })
        
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        rows, cols = 7000, 6000 
        return {
            "input": torch.empty(rows, cols, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "output": torch.zeros(cols, rows, device="cuda", dtype=dtype),
            "rows": rows,
            "cols": cols
        } 