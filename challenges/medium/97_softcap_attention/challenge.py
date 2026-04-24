import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Softcap Attention",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        N: int,
        d_model: int,
        h: int,
        softcap: float,
    ):
        assert Q.shape == (N, d_model)
        assert K.shape == (N, d_model)
        assert V.shape == (N, d_model)
        assert output.shape == (N, d_model)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32
        assert Q.device.type == "cuda"
        assert K.device.type == "cuda"
        assert V.device.type == "cuda"
        assert output.device.type == "cuda"
        assert d_model % h == 0

        d_k = d_model // h
        scale = 1.0 / math.sqrt(d_k)
        result = torch.zeros((N, d_model), dtype=Q.dtype, device=Q.device)
        for head in range(h):
            Q_h = Q[:, head * d_k : (head + 1) * d_k]
            K_h = K[:, head * d_k : (head + 1) * d_k]
            V_h = V[:, head * d_k : (head + 1) * d_k]
            scores = torch.matmul(Q_h, K_h.t()) * scale
            scores = softcap * torch.tanh(scores / softcap)
            attn = torch.softmax(scores, dim=1)
            head_output = torch.matmul(attn, V_h)
            result[:, head * d_k : (head + 1) * d_k] = head_output
        output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "h": (ctypes.c_int, "in"),
            "softcap": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor([[1.0, 0.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], device="cuda", dtype=dtype)
        K = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device="cuda", dtype=dtype)
        V = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]], device="cuda", dtype=dtype)
        output = torch.empty(2, 4, device="cuda", dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "N": 2,
            "d_model": 4,
            "h": 2,
            "softcap": 5.0,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(0)
        dtype = torch.float32
        tests = []

        # example
        tests.append(self.generate_example_test())

        # single head, tiny
        Q = torch.tensor([[1.0, 1.0], [2.0, 2.0]], device="cuda", dtype=dtype)
        K = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device="cuda", dtype=dtype)
        V = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device="cuda", dtype=dtype)
        output = torch.empty(2, 2, device="cuda", dtype=dtype)
        tests.append(
            {
                "Q": Q,
                "K": K,
                "V": V,
                "output": output,
                "N": 2,
                "d_model": 2,
                "h": 1,
                "softcap": 10.0,
            }
        )

        # zero inputs
        tests.append(
            {
                "Q": torch.zeros(4, 8, device="cuda", dtype=dtype),
                "K": torch.zeros(4, 8, device="cuda", dtype=dtype),
                "V": torch.zeros(4, 8, device="cuda", dtype=dtype),
                "output": torch.empty(4, 8, device="cuda", dtype=dtype),
                "N": 4,
                "d_model": 8,
                "h": 2,
                "softcap": 5.0,
            }
        )

        # mixed negative values, strong softcap (heavy clipping)
        tests.append(
            {
                "Q": torch.empty(4, 8, device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "K": torch.empty(4, 8, device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "V": torch.empty(4, 8, device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty(4, 8, device="cuda", dtype=dtype),
                "N": 4,
                "d_model": 8,
                "h": 4,
                "softcap": 1.0,
            }
        )

        # power-of-2, many heads
        tests.append(
            {
                "Q": torch.empty(32, 32, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "K": torch.empty(32, 32, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "V": torch.empty(32, 32, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(32, 32, device="cuda", dtype=dtype),
                "N": 32,
                "d_model": 32,
                "h": 8,
                "softcap": 20.0,
            }
        )

        # power-of-2, larger head dim
        tests.append(
            {
                "Q": torch.empty(64, 128, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
                "K": torch.empty(64, 128, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
                "V": torch.empty(64, 128, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
                "output": torch.empty(64, 128, device="cuda", dtype=dtype),
                "N": 64,
                "d_model": 128,
                "h": 8,
                "softcap": 30.0,
            }
        )

        # non-power-of-2 sequence length
        tests.append(
            {
                "Q": torch.empty(30, 64, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "K": torch.empty(30, 64, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "V": torch.empty(30, 64, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(30, 64, device="cuda", dtype=dtype),
                "N": 30,
                "d_model": 64,
                "h": 8,
                "softcap": 15.0,
            }
        )

        # non-power-of-2 sequence length, larger
        tests.append(
            {
                "Q": torch.empty(255, 64, device="cuda", dtype=dtype).uniform_(-3.0, 3.0),
                "K": torch.empty(255, 64, device="cuda", dtype=dtype).uniform_(-3.0, 3.0),
                "V": torch.empty(255, 64, device="cuda", dtype=dtype).uniform_(-3.0, 3.0),
                "output": torch.empty(255, 64, device="cuda", dtype=dtype),
                "N": 255,
                "d_model": 64,
                "h": 4,
                "softcap": 50.0,
            }
        )

        # large softcap (approaches plain attention)
        tests.append(
            {
                "Q": torch.empty(128, 128, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "K": torch.empty(128, 128, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "V": torch.empty(128, 128, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(128, 128, device="cuda", dtype=dtype),
                "N": 128,
                "d_model": 128,
                "h": 8,
                "softcap": 1000.0,
            }
        )

        # realistic inference-size case
        tests.append(
            {
                "Q": torch.empty(512, 256, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "K": torch.empty(512, 256, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "V": torch.empty(512, 256, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(512, 256, device="cuda", dtype=dtype),
                "N": 512,
                "d_model": 256,
                "h": 8,
                "softcap": 50.0,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, d_model, h = 2048, 1024, 16
        Q = torch.empty(N, d_model, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        K = torch.empty(N, d_model, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        V = torch.empty(N, d_model, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        output = torch.empty(N, d_model, device="cuda", dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "N": N,
            "d_model": d_model,
            "h": h,
            "softcap": 50.0,
        }
