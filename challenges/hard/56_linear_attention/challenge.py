import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Linear Self-Attention",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free"
        )
    
    def reference_impl(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
        assert Q.shape == K.shape == V.shape == output.shape == (M,d)
        # φ(x) = ELU(x) + 1
        phi_Q = torch.where(Q > 0, Q + 1, torch.exp(Q))
        phi_K = torch.where(K > 0, K + 1, torch.exp(K))

        # S = sum_j φ(K_j) V_j^T = φ(K)^T V
        S = phi_K.T @ V         # (d,M) @ (M,d) = (d, d)
        # z = sum_j φ(K_j)
        z = phi_K.sum(dim=0)    # (d,)

        # numerator: φ(Q_i) @ S  → (M,d)
        numerator = phi_Q @ S   # (M,d) @ (d,d) = (M,d)
        # denominator: φ(Q_i) @ z  → (scalar)
        denominator = phi_Q @ z # (M,d) @ (d,) = (M,)
        
        output.copy_(numerator / denominator.unsqueeze(-1)) # (M, d)


    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype)
        K= torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype)
        V= torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device="cuda", dtype=dtype)
        output = torch.empty(2, 4, device="cuda", dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": 2, "d": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_example
        tests.append({
            "Q": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype),
            "K": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype),
            "V": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device="cuda", dtype=dtype),
            "output": torch.empty(2, 4, device="cuda", dtype=dtype),
            "M": 2, "d": 4
        })

        # basic_example
        tests.append({
            "Q": torch.tensor([[0.0,0.0], [1.0, 1.0]], device="cuda", dtype=dtype),
            "K": torch.tensor([[1.0,0.0], [0.0, 1.0]], device="cuda", dtype=dtype),
            "V": torch.tensor([[3.0, 4.0], [5.0, 6.0]], device="cuda", dtype=dtype),
            "output": torch.empty(2, 2, device="cuda", dtype=dtype),
            "M": 2, "d": 2
        })

        # zero_matrices
        tests.append({
            "Q": torch.zeros((3, 5), device="cuda", dtype=dtype),
            "K": torch.zeros((3, 5), device="cuda", dtype=dtype),
            "V": torch.zeros((3, 5), device="cuda", dtype=dtype),
            "output": torch.empty(3, 5, device="cuda", dtype=dtype),
            "M": 3, "d": 5
        })

        # mixed_values
        tests.append({
            "Q": torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]], device="cuda", dtype=dtype),
            "K": torch.tensor([[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]], device="cuda", dtype=dtype),
            "V": torch.tensor([[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]], device="cuda", dtype=dtype),
            "output": torch.empty(4, 3, device="cuda", dtype=dtype),
            "M": 4, "d": 3
        })

        # large_matrices
        tests.append({
            "Q": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "K": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "V": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "output": torch.empty(128, 32, device="cuda", dtype=dtype),
            "M": 128, "d": 32
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, d = 10000, 128
        Q = torch.empty((M, d), device="cuda", dtype=dtype).uniform_(-100, 100)
        K = torch.empty((M, d), device="cuda", dtype=dtype).uniform_(-100, 100)
        V = torch.empty((M, d), device="cuda", dtype=dtype).uniform_(-100, 100)
        output = torch.empty(M, d, device="cuda", dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": M, "d": d}
