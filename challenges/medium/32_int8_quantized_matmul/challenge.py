import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="INT8 Quantized MatMul",
            atol=0,
            rtol=0,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, scale_A: float, scale_B: float, scale_C: float, zero_point_A: int, zero_point_B: int, zero_point_C: int):
        A = A.view(M, K).to(torch.int32)
        B = B.view(K, N).to(torch.int32)
        A_f = (A - zero_point_A).to(torch.float32)
        B_f = (B - zero_point_B).to(torch.float32)
        C_f = torch.matmul(A_f, B_f).round().int() # closest thing to integer accumulation we have
        C_f = C_f * scale_A * scale_B / scale_C
        C_q = torch.round(C_f).to(torch.int32) + zero_point_C
        C_q = torch.clamp(C_q, -128, 127).to(torch.int8)
        C.view(M, N).copy_(C_q)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_int8), "in"),
            "B": (ctypes.POINTER(ctypes.c_int8), "in"),
            "C": (ctypes.POINTER(ctypes.c_int8), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "scale_A": (ctypes.c_float, "in"),
            "scale_B": (ctypes.c_float, "in"),
            "scale_C": (ctypes.c_float, "in"),
            "zero_point_A": (ctypes.c_int, "in"),
            "zero_point_B": (ctypes.c_int, "in"),
            "zero_point_C": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int8
        device = "cuda"
        A = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device).flatten()
        B = torch.tensor([[5, 6], [7, 8]], dtype=dtype, device=device).flatten()
        C = torch.tensor([[0, 0], [0, 0]], dtype=dtype, device=device).flatten()

        return {
            "A": A,
            "B": B,
            "C": C,
            "M": 2,
            "N": 2,
            "K": 2,
            "scale_A": 0.1,
            "scale_B": 0.2,
            "scale_C": 0.05,
            "zero_point_A": 0,
            "zero_point_B": 0,
            "zero_point_C": 0
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int8
        device = "cuda"
        tests = []

        # 1. 4x4x4_zero_zp
        A1 = torch.randint(-128, 128, (4 ,4), dtype=dtype, device=device)
        B1 = torch.randint(-128, 128, (4 ,4), dtype=dtype, device=device)
        C1 = torch.randint(-128, 128, (4, 4), dtype=dtype, device=device)
        tests.append({
            "A": A1, "B": B1, "C": C1, "M": 4, "N": 4, "K": 4,
            "scale_A": 0.1, "scale_B": 0.2, "scale_C": 0.05,
            "zero_point_A": 0, "zero_point_B": 0, "zero_point_C": 0
        })

        # 2. 2x3x5_nonzero_zp
        A2 = torch.randint(-128, 128, (2 , 5), dtype=dtype, device=device)
        B2 = torch.randint(-128, 128, (5, 3), dtype=dtype, device=device)
        C2 = torch.empty((2, 3), dtype=dtype, device=device)
        tests.append({
            "A": A2, "B": B2, "C": C2, "M": 2, "N": 3, "K": 5,
            "scale_A": 0.5, "scale_B": 0.25, "scale_C": 0.125,
            "zero_point_A": 1, "zero_point_B": -2, "zero_point_C": 3
        })

        # 3. 1x1x3
        A3 = torch.randint(-128, 128, (1, 3), dtype=dtype, device=device)
        B3 = torch.randint(-128, 128, (3, 1), dtype=dtype, device=device)
        C3 = torch.empty((1, 1), dtype=dtype, device=device)
        tests.append({
            "A": A3, "B": B3, "C": C3, "M": 1, "N": 1, "K": 3,
            "scale_A": 1.0, "scale_B": 1.0, "scale_C": 1.0,
            "zero_point_A": 1, "zero_point_B": 3, "zero_point_C": 5
        })

        # 4. 3x5x2
        A4 = torch.randint(-50, 51, (3, 2), dtype=dtype, device=device)
        B4 = torch.randint(-50, 51, (2, 5), dtype=dtype, device=device)
        C4 = torch.zeros((3, 5), dtype=dtype, device=device)
        tests.append({
            "A": A4, "B": B4, "C": C4, "M": 3, "N": 5, "K": 2,
            "scale_A": 0.05, "scale_B": 0.1, "scale_C": 0.01,
            "zero_point_A": 0, "zero_point_B": 0, "zero_point_C": 0
        })

        # 5. 32x32x16
        A5 = torch.randint(-128, 128, (32, 16), dtype=dtype, device=device)
        B5 = torch.randint(-128, 128, (16, 32), dtype=dtype, device=device)
        C5 = torch.empty((32, 32), dtype=dtype, device=device)
        tests.append({
            "A": A5, "B": B5, "C": C5, "M": 32, "N": 32, "K": 16,
            "scale_A": 0.2, "scale_B": 0.3, "scale_C": 0.1,
            "zero_point_A": 0, "zero_point_B": 0, "zero_point_C": 0
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int8
        device = "cuda"
        shape_A = (8192, 2048)
        shape_B = (2048, 4096)
        shape_C = (8192, 4096)
        A = torch.randint(-128, 128, (shape_A[0] * shape_A[1],), dtype=dtype, device=device)
        B = torch.randint(-128, 128, (shape_B[0] * shape_B[1],), dtype=dtype, device=device)
        C = torch.empty(shape_C[0] * shape_C[1], dtype=dtype, device=device)
        M = 8192
        N = 4096
        K = 2048
        scale_A = 0.1
        scale_B = 0.1
        scale_C = 0.01
        zero_point_A = 0
        zero_point_B = 0
        zero_point_C = 0
        return {
            "A": A,
            "B": B,
            "C": C,
            "M": M,
            "N": N,
            "K": K,
            "scale_A": scale_A,
            "scale_B": scale_B,
            "scale_C": scale_C,
            "zero_point_A": zero_point_A,
            "zero_point_B": zero_point_B,
            "zero_point_C": zero_point_C
        }
