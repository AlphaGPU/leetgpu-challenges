import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="General Matrix Multiplication (GEMM)",
            atol=5e-2,
            rtol=5e-2,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, alpha: float, beta: float):
        assert A.shape == (M , K)
        assert B.shape == (K ,N)
        assert C.shape == (M , N)
        A_f32 = A.view(M, K).to(torch.float32)
        B_f32 = B.view(K, N).to(torch.float32)
        C_f32 = C.view(M, N).to(torch.float32)
        matmul_result = torch.matmul(A_f32, B_f32)
        final_result = alpha * matmul_result + beta * C_f32
        C.copy_(final_result.to(torch.float16))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "B": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "C": (ctypes.POINTER(ctypes.c_uint16), "inout"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "alpha": (ctypes.c_float, "in"),
            "beta": (ctypes.c_float, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float16
        A = torch.tensor([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]], device="cuda", dtype=dtype)
        B = torch.tensor([[1.0, 2.0],[3.0, 4.0],[5.0, 6.0]], device="cuda", dtype=dtype)
        C = torch.tensor([[1.0, 1.0],[1.0, 1.0]], device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "M": 2,
            "N": 2,
            "K": 3,
            "alpha": 1.0,
            "beta": 0.0
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float16
        tests = []

        # 16x16x16_a1_b0
        tests.append({
            "A": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "B": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "C": torch.zeros((16, 16), device="cuda", dtype=dtype),
            "M": 16,
            "N": 16,
            "K": 16,
            "alpha": 1.0,
            "beta": 0.0
        })

        # 16x16x16_a1_b1
        tests.append({
            "A": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-0.5, 0.5),
            "B": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-0.5, 0.5),
            "C": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-0.5, 0.5),
            "M": 16,
            "N": 16,
            "K": 16,
            "alpha": 1.0,
            "beta": 1.0
        })

        # 32x16x16_a0.5_b0.5
        tests.append({
            "A": torch.empty((32, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "B": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "C": torch.empty((32, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "M": 32,
            "N": 16,
            "K": 16,
            "alpha": 0.5,
            "beta": 0.5
        })

        # 16x32x16_a1_b1
        tests.append({
            "A": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "B": torch.empty((16, 32), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "C": torch.empty((16, 32), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "M": 16,
            "N": 32,
            "K": 16,
            "alpha": 1.0,
            "beta": 1.0
        })

        # 16x16x32_a0_b1
        tests.append({
            "A": torch.empty((16, 32), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "B": torch.empty((32, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "C": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "M": 16,
            "N": 16,
            "K": 32,
            "alpha": 0.0,
            "beta": 1.0
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float16
        M = 1024
        N = 1024
        K = 1024
        A = torch.empty((M, K), device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        B = torch.empty((K, N), device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        C = torch.empty((M, N), device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        return {
            "A": A,
            "B": B,
            "C": C,
            "M": M,
            "N": N,
            "K": K,
            "alpha": 1.0,
            "beta": 1.0
        }
