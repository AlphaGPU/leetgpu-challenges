import ctypes
from typing import Any, List, Dict
import torch
import numpy as np
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Radix Sort",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
       
        assert input.dtype == torch.uint32
        assert output.dtype == torch.uint32
        assert input.shape == output.shape == (N,)
        
        # Convert uint32 to int64 for sorting (since torch.sort doesn't support uint32)
        input_int64 = input.to(torch.int64)
        sorted_tensor = torch.sort(input_int64)[0]
        # Convert back to uint32
        output.copy_(sorted_tensor.to(torch.uint32))
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_uint32), "in"),
            "output": (ctypes.POINTER(ctypes.c_uint32), "out"),
            "N": (ctypes.c_int, "in")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.uint32
        N = 8
        input_data = torch.tensor([170, 45, 75, 90, 2, 802, 24, 66], device="cuda", dtype=dtype)
        output_data = torch.zeros(N, device="cuda", dtype=dtype)
        return {
            "input": input_data,
            "output": output_data,
            "N": N
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.uint32
        test_cases = []
        
        # Test case 1: basic example
        test_cases.append({
            "input": torch.tensor([170, 45, 75, 90, 2, 802, 24, 66], device="cuda", dtype=dtype),
            "output": torch.zeros(8, device="cuda", dtype=dtype),
            "N": 8
        })
        
        # Test case 2: duplicate numbers
        test_cases.append({
            "input": torch.tensor([1, 4, 1, 3, 555, 1000, 2], device="cuda", dtype=dtype),
            "output": torch.zeros(7, device="cuda", dtype=dtype),
            "N": 7
        })
        
        # Test case 3: single element
        test_cases.append({
            "input": torch.tensor([42], device="cuda", dtype=dtype),
            "output": torch.zeros(1, device="cuda", dtype=dtype),
            "N": 1
        })
        
        # Test case 4: already sorted
        test_cases.append({
            "input": torch.tensor([1, 2, 3, 4, 5, 6], device="cuda", dtype=dtype),
            "output": torch.zeros(6, device="cuda", dtype=dtype),
            "N": 6
        })
        
        # Test case 5: reverse sorted
        test_cases.append({
            "input": torch.tensor([6, 5, 4, 3, 2, 1], device="cuda", dtype=dtype),
            "output": torch.zeros(6, device="cuda", dtype=dtype),
            "N": 6
        })
        
        # Test case 6: large numbers
        test_cases.append({
            "input": torch.tensor([4294967295, 1000000000, 500000000, 2000000000, 100000000], device="cuda", dtype=dtype),
            "output": torch.zeros(5, device="cuda", dtype=dtype),
            "N": 5
        })
        
        # Test case 7: medium random
        test_cases.append({
            "input": torch.randint(0, 1000001, (1024,), device="cuda", dtype=dtype),
            "output": torch.zeros(1024, device="cuda", dtype=dtype),
            "N": 1024
        })
        
        # Test case 8: large random
        test_cases.append({
            "input": torch.randint(0, 4294967296, (10000,), device="cuda", dtype=dtype),
            "output": torch.zeros(10000, device="cuda", dtype=dtype),
            "N": 10000
        })
        
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.uint32
        N = 50000000
        return {
            "input": torch.randint(0, 4294967296, (N,), device="cuda", dtype=dtype),
            "output": torch.zeros(N, device="cuda", dtype=dtype),
            "N": N
        } 