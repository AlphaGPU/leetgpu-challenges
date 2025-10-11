import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="2D Max Pooling",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, 
                      N: int, C: int, H: int, W: int,
                      kernel_size: int, stride: int, padding: int):
        input_tensor = input.view(N, C, H, W)
        
        # Apply max pooling
        result = torch.nn.functional.max_pool2d(
            input_tensor, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
    
        output.copy_(result.flatten())
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "W": (ctypes.c_int, "in"),
            "kernel_size": (ctypes.c_int, "in"),
            "stride": (ctypes.c_int, "in"),
            "padding": (ctypes.c_int, "in")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        """Simple test case matching the example in challenge.html"""
        dtype = torch.float32
        N, C, H, W = 1, 1, 3, 3
        kernel_size, stride, padding = 2, 1, 0
        
        # Create input tensor: [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
        input_tensor = torch.tensor([[[[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]]]], device="cuda", dtype=dtype)
        
        # Calculate output dimensions
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        return {
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N,
            "C": C,
            "H": H,
            "W": W,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """Comprehensive test suite covering various scenarios and edge cases"""
        dtype = torch.float32
        test_cases = []
        
        # Set seed for reproducible random tests
        torch.manual_seed(42)
        
        # Test case 1: 2x2 kernel, stride 2, no padding (deterministic)
        N, C, H, W = 1, 1, 4, 4
        kernel_size, stride, padding = 2, 2, 0
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        input_tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0],
                                      [9.0, 10.0, 11.0, 12.0],
                                      [13.0, 14.0, 15.0, 16.0]]]], device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        # Test case 2: 3x3 kernel, stride 1, padding 1 (random data)
        N, C, H, W = 1, 2, 5, 5
        kernel_size, stride, padding = 3, 1, 1
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        input_tensor = torch.randn(N, C, H, W, device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        # Test case 3: 1x1 kernel, stride 1, no padding (identity operation)
        N, C, H, W = 2, 3, 8, 8
        kernel_size, stride, padding = 1, 1, 0
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        input_tensor = torch.randn(N, C, H, W, device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        # Test case 4: Large kernel with padding
        N, C, H, W = 1, 1, 10, 10
        kernel_size, stride, padding = 5, 2, 2
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        input_tensor = torch.randn(N, C, H, W, device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        # Test case 5: Edge case with small dimensions
        N, C, H, W = 1, 1, 2, 2
        kernel_size, stride, padding = 2, 1, 0
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        # Test case 6: Boundary conditions - kernel size equals input size
        N, C, H, W = 1, 1, 3, 3
        kernel_size, stride, padding = 3, 1, 0
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        input_tensor = torch.tensor([[[[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]]]], device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        # Test case 7: Large padding relative to input size
        N, C, H, W = 1, 1, 4, 4
        kernel_size, stride, padding = 2, 1, 1
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        input_tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0],
                                      [9.0, 10.0, 11.0, 12.0],
                                      [13.0, 14.0, 15.0, 16.0]]]], device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        # Test case 8: Multiple channels with different patterns
        N, C, H, W = 1, 3, 6, 6
        kernel_size, stride, padding = 2, 2, 1
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        # Create structured input with different patterns per channel
        input_tensor = torch.zeros(N, C, H, W, device="cuda", dtype=dtype)
        input_tensor[0, 0, :, :] = torch.arange(H * W, device="cuda", dtype=dtype).reshape(H, W)
        input_tensor[0, 1, :, :] = torch.arange(H * W, device="cuda", dtype=dtype).reshape(H, W).flip(0)
        input_tensor[0, 2, :, :] = torch.arange(H * W, device="cuda", dtype=dtype).reshape(H, W).flip(1)
        
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        # Test case 9: Extreme values and edge cases
        N, C, H, W = 1, 1, 5, 5
        kernel_size, stride, padding = 2, 1, 0
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        # Create input with extreme values
        input_tensor = torch.tensor([[[[1e6, -1e6, 0.0, 1e-6, -1e-6],
                                      [float('inf'), float('-inf'), 1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0, 7.0, 8.0],
                                      [9.0, 10.0, 11.0, 12.0, 13.0],
                                      [14.0, 15.0, 16.0, 17.0, 18.0]]]], device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        # Test case 10: Non-power-of-two dimensions
        N, C, H, W = 1, 1, 7, 11
        kernel_size, stride, padding = 3, 2, 1
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        input_tensor = torch.randn(N, C, H, W, device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        test_cases.append({
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        })
        
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        """Large test case for performance evaluation"""
        dtype = torch.float32
        # Reasonable size for performance testing without memory issues
        N, C, H, W = 4, 64, 256, 256  # 4 batches, 64 channels, 256x256 spatial
        kernel_size, stride, padding = 3, 2, 1
        
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
        
        # Use seeded random for reproducible performance tests
        torch.manual_seed(123)
        input_tensor = torch.randn(N, C, H, W, device="cuda", dtype=dtype)
        output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)
        
        return {
            "input": input_tensor.flatten(),
            "output": output_tensor,
            "N": N, "C": C, "H": H, "W": W,
            "kernel_size": kernel_size, "stride": stride, "padding": padding
        }
