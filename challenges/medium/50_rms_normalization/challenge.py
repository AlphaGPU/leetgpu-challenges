import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="RMS Normalization",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, input: torch.Tensor, gamma: float, beta: float, 
                      output: torch.Tensor, N: int, eps: float):
        assert input.shape == output.shape == (N,)
        assert input.dtype == output.dtype
        assert input.device == output.device

        # RMSNorm: compute root mean square (without mean-centering)
        rms = torch.sqrt(torch.mean(input**2) + eps)  # shape: scalar
        
        # Normalize
        normalized = input / rms  # shape: [N]
        
        # Scale and shift
        output.copy_(gamma * normalized + beta)
        
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.c_float, "in"),    
            "beta": (ctypes.c_float, "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        input = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        gamma = 1.0
        beta = 0.0
        output = torch.empty(N, device="cuda", dtype=dtype)
        eps = 1e-5
        return {
            "input": input,
            "gamma": gamma,
            "beta": beta,
            "output": output,
            "N": N,
            "eps": eps
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_small
        N = 3
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype),
            "gamma": 1.0,
            "beta": 0.0,
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N, "eps": 1e-5
        })
        
        # single_feature
        N = 1
        tests.append({
            "input": torch.tensor([5.0], device="cuda", dtype=dtype),
            "gamma": 2.0,
            "beta": -1.0,
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N, "eps": 1e-5
        })

        # all zeros
        N = 4
        tests.append({
            "input": torch.zeros(N, device="cuda", dtype=dtype),
            "gamma": 1.0,
            "beta": 0.0,
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N, "eps": 1e-5
        })
        
        # negative numbers
        N = 5
        tests.append({
            "input": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], device="cuda", dtype=dtype),
            "gamma": 1.0,
            "beta": 0.0,
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N, "eps": 1e-5
        })

        # different gamma/beta
        N = 3
        tests.append({
            "input": torch.tensor([0.0, 1.0, 2.0], device="cuda", dtype=dtype),
            "gamma": 0.5,
            "beta": -1.0,
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N, "eps": 1e-5
        })
        
        # large values
        N = 8
        tests.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "gamma": 1.5,
            "beta": 0.0,
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N, "eps": 1e-5
        })

    
        # large N
        N = 2000
        tests.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "gamma": 1.3,
            "beta": 0.0,
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N, "eps": 1e-5
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 100000
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "gamma": 1.5,
            "beta": 0.0,
            "output": torch.empty(N, device="cuda", dtype=dtype),
            "N": N,
            "eps": 1e-5
        }