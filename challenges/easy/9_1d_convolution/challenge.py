import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="1D Convolution",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
        assert input.shape == (input_size,)
        assert kernel.shape == (kernel_size,)
        assert output.shape == (input_size - kernel_size + 1,)
        assert input.dtype == kernel.dtype == output.dtype
        assert input.device == kernel.device == output.device
        
        # Create strided view of input for all windows
        windows = input.unfold(0, kernel_size, 1)
        
        # Use einsum for explicit cross-correlation
        # 'ij,j->i' means: for each window i, multiply with kernel j and sum over j
        output.copy_(torch.einsum('ij,j->i', windows, kernel))
            
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "kernel": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "input_size": (ctypes.c_int, "in"),
            "kernel_size": (ctypes.c_int, "in")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=dtype)
        kernel_tensor = torch.tensor([1.0, 0.0, -1.0], device="cuda", dtype=dtype)
        output_tensor = torch.empty(3, device="cuda", dtype=dtype)
        return {
            "input": input_tensor,
            "kernel": kernel_tensor,
            "output": output_tensor,
            "input_size": 5,
            "kernel_size": 3
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_specs = [
            # Basic test cases
            ("basic_5x3", [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 0.0, -1.0]),
            ("basic_4x2", [2.0, 4.0, 6.0, 8.0], [0.5, 0.2]),
            ("identity_kernel", [1.0, 2.0, 3.0, 4.0], [1.0]),
            ("edge_detection", [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0, -1.0]),
            ("smoothing", [1.0, 2.0, 3.0, 4.0, 5.0], [0.25, 0.5, 0.25]),
        ]
        
        test_cases = []
        for _, input_vals, kernel_vals in test_specs:
            input_size = len(input_vals)
            kernel_size = len(kernel_vals)
            output_size = input_size - kernel_size + 1
            test_cases.append({
                "input": torch.tensor(input_vals, device="cuda", dtype=dtype),
                "kernel": torch.tensor(kernel_vals, device="cuda", dtype=dtype),
                "output": torch.empty(output_size, device="cuda", dtype=dtype),
                "input_size": input_size,
                "kernel_size": kernel_size
            })
        
        # Random test cases with different sizes
        for _, input_size, kernel_size in [
            ("small_conv", 10, 3),
            ("medium_conv", 100, 7),
            ("large_conv", 1000, 15),
            ("wide_kernel", 50, 20),
            ("narrow_kernel", 200, 2),
        ]:
            output_size = input_size - kernel_size + 1
            test_cases.append({
                "input": torch.empty(input_size, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "kernel": torch.empty(kernel_size, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(output_size, device="cuda", dtype=dtype),
                "input_size": input_size,
                "kernel_size": kernel_size
            })
        
        # Edge cases
        for _, input_size, kernel_size in [
            ("min_input", 1, 1),
            ("kernel_equals_input", 10, 10),
            ("large_input_small_kernel", 10000, 3),
        ]:
            output_size = input_size - kernel_size + 1
            test_cases.append({
                "input": torch.empty(input_size, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "kernel": torch.empty(kernel_size, device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
                "output": torch.empty(output_size, device="cuda", dtype=dtype),
                "input_size": input_size,
                "kernel_size": kernel_size
            })
        
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input_size, kernel_size = 1500000, 2047  # Large convolution for performance testing
        output_size = input_size - kernel_size + 1
        return {
            "input": torch.empty(input_size, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "kernel": torch.empty(kernel_size, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "output": torch.empty(output_size, device="cuda", dtype=dtype),
            "input_size": input_size,
            "kernel_size": kernel_size
        } 