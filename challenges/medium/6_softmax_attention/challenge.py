import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Softmax Attention",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):
        scale = d ** 0.5
        attn = torch.matmul(Q, K.t()) / scale
        attn = torch.softmax(attn, dim=1)
        torch.matmul(attn, V, out=output)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype)
        K = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device="cuda", dtype=dtype)
        V = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], device="cuda", dtype=dtype)
        output = torch.empty(2, 4, device="cuda", dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": 2, "N": 3, "d": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_example
        tests.append({
            "Q": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype),
            "K": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device="cuda", dtype=dtype),
            "V": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], device="cuda", dtype=dtype),
            "output": torch.empty(2, 4, device="cuda", dtype=dtype),
            "M": 2, "N": 3, "d": 4
        })

        # zero_matrices
        tests.append({
            "Q": torch.zeros((3, 5), device="cuda", dtype=dtype),
            "K": torch.zeros((3, 5), device="cuda", dtype=dtype),
            "V": torch.zeros((3, 5), device="cuda", dtype=dtype),
            "output": torch.empty(3, 5, device="cuda", dtype=dtype),
            "M": 3, "N": 3, "d": 5
        })

        # mixed_values
        tests.append({
            "Q": torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]], device="cuda", dtype=dtype),
            "K": torch.tensor([[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]], device="cuda", dtype=dtype),
            "V": torch.tensor([[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]], device="cuda", dtype=dtype),
            "output": torch.empty(4, 3, device="cuda", dtype=dtype),
            "M": 4, "N": 4, "d": 3
        })

        # large_matrices
        tests.append({
            "Q": torch.empty((64, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "K": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "V": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "output": torch.empty(64, 32, device="cuda", dtype=dtype),
            "M": 64, "N": 128, "d": 32
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N, d = 512, 256, 128
        Q = torch.empty((512, 128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
        K = torch.empty((256, 128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
        V = torch.empty((256,128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
        output = torch.empty(M, d, device="cuda", dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": M, "N": N, "d": d}
