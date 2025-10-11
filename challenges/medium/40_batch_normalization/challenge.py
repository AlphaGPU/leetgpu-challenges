import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Batch Normalization",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, 
                      output: torch.Tensor, N: int, C: int, eps: float):
        assert input.shape == output.shape == (N, C)
        assert gamma.shape == beta.shape == (C,)
        assert input.dtype == gamma.dtype == beta.dtype == output.dtype
        assert input.device == gamma.device == beta.device == output.device
        
        # Compute mean and variance for each feature channel
        mean = torch.mean(input, dim=0)  # Shape: [C]
        variance = torch.var(input, dim=0, unbiased=False)  # Shape: [C]
        
        # Normalize
        normalized = (input - mean) / torch.sqrt(variance + eps)
        
        # Scale and shift
        output.copy_(gamma * normalized + beta)
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in")
        }
        
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, C = 3, 2
        input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda", dtype=dtype)
        gamma = torch.tensor([1.0, 1.0], device="cuda", dtype=dtype)
        beta = torch.tensor([0.0, 0.0], device="cuda", dtype=dtype)
        output = torch.empty((N, C), device="cuda", dtype=dtype)
        eps = 1e-5
        return {
            "input": input,
            "gamma": gamma,
            "beta": beta,
            "output": output,
            "N": N,
            "C": C,
            "eps": eps
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        
        # basic_small
        N, C = 3, 2
        tests.append({
            "input": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda", dtype=dtype),
            "gamma": torch.tensor([1.0, 1.0], device="cuda", dtype=dtype),
            "beta": torch.tensor([0.0, 0.0], device="cuda", dtype=dtype),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N, "C": C, "eps": 1e-5
        })
        
        # single_batch
        N, C = 1, 4
        tests.append({
            "input": torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda", dtype=dtype),
            "gamma": torch.tensor([1.0, 1.0, 1.0, 1.0], device="cuda", dtype=dtype),
            "beta": torch.tensor([0.0, 0.0, 0.0, 0.0], device="cuda", dtype=dtype),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N, "C": C, "eps": 1e-5
        })
        
        # all_zeros
        N, C = 4, 3
        tests.append({
            "input": torch.zeros((N, C), device="cuda", dtype=dtype),
            "gamma": torch.ones(C, device="cuda", dtype=dtype),
            "beta": torch.zeros(C, device="cuda", dtype=dtype),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N, "C": C, "eps": 1e-5
        })
        
        # negative_numbers
        N, C = 2, 3
        tests.append({
            "input": torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], device="cuda", dtype=dtype),
            "gamma": torch.tensor([1.0, 1.0, 1.0], device="cuda", dtype=dtype),
            "beta": torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=dtype),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N, "C": C, "eps": 1e-5
        })
        
        # different_gamma_beta
        N, C = 2, 2
        tests.append({
            "input": torch.tensor([[0.0, 1.0], [2.0, 3.0]], device="cuda", dtype=dtype),
            "gamma": torch.tensor([2.0, 0.5], device="cuda", dtype=dtype),
            "beta": torch.tensor([1.0, -1.0], device="cuda", dtype=dtype),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N, "C": C, "eps": 1e-5
        })
        
        # large_values
        N, C = 5, 3
        tests.append({
            "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-50.0, 50.0),
            "gamma": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.5, 2.0),
            "beta": torch.empty(C, device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N, "C": C, "eps": 1e-5
        })
        
        # medium_size
        N, C = 64, 32
        tests.append({
            "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "gamma": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.5, 2.0),
            "beta": torch.empty(C, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N, "C": C, "eps": 1e-5
        })
        
        # single_feature
        N, C = 100, 1
        tests.append({
            "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "gamma": torch.tensor([1.5], device="cuda", dtype=dtype),
            "beta": torch.tensor([0.5], device="cuda", dtype=dtype),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N, "C": C, "eps": 1e-5
        })
        
        # high_variance
        N, C = 10, 5
        input_data = torch.empty((N, C), device="cuda", dtype=dtype)
        for i in range(C):
            input_data[:, i] = torch.linspace(-100 + i*10, 100 - i*10, N, device="cuda", dtype=dtype)
        tests.append({
            "input": input_data,
            "gamma": torch.ones(C, device="cuda", dtype=dtype),
            "beta": torch.zeros(C, device="cuda", dtype=dtype),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N, "C": C, "eps": 1e-5
        })
        
        return tests
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, C = 5000, 512
        return {
            "input": torch.empty((N, C), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "gamma": torch.empty(C, device="cuda", dtype=dtype).uniform_(0.5, 2.0),
            "beta": torch.empty(C, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
            "output": torch.empty((N, C), device="cuda", dtype=dtype),
            "N": N,
            "C": C,
            "eps": 1e-5
        }
