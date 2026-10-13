import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused LayerNorm Linear GELU"
    # Outputs are O(1); float32 error from the normalization and a K <= 4,096 dot product is
    # ~1e-5. Skipping gamma/beta, the bias, the variance epsilon, or using the tanh GELU
    # approximation shifts outputs by >= 1e-3 on the targeted cases.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        W: torch.Tensor,
        bias: torch.Tensor,
        y: torch.Tensor,
        M: int,
        K: int,
        N: int,
        eps: float,
    ):
        assert x.shape == (M, K)
        assert gamma.shape == (K,)
        assert beta.shape == (K,)
        assert W.shape == (K, N)
        assert bias.shape == (N,)
        assert y.shape == (M, N)
        assert x.dtype == gamma.dtype == beta.dtype == W.dtype == torch.float32
        assert bias.dtype == y.dtype == torch.float32

        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        xn = (x - mean) / torch.sqrt(var + eps) * gamma + beta
        h = xn @ W + bias
        y.copy_(0.5 * h * (1.0 + torch.erf(h / (2.0**0.5))))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "W": (ctypes.POINTER(ctypes.c_float), "in"),
            "bias": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def _make(self, M, K, N, eps=1e-5, seed=0, x_range=(-3.0, 3.0)):
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        return {
            "x": torch.empty(M, K, device=dev, dtype=f32).uniform_(*x_range),
            "gamma": torch.empty(K, device=dev, dtype=f32).uniform_(0.5, 1.5),
            "beta": torch.empty(K, device=dev, dtype=f32).uniform_(-0.5, 0.5),
            "W": torch.empty(K, N, device=dev, dtype=f32).uniform_(-1.0, 1.0) / (K**0.5),
            "bias": torch.empty(N, device=dev, dtype=f32).uniform_(-0.5, 0.5),
            "y": torch.zeros(M, N, device=dev, dtype=f32),
            "M": M,
            "K": K,
            "N": N,
            "eps": eps,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "x": torch.tensor([[1.0, 3.0], [2.0, 2.0]], device=dev, dtype=f32),
            "gamma": torch.tensor([1.0, 1.0], device=dev, dtype=f32),
            "beta": torch.tensor([0.0, 0.0], device=dev, dtype=f32),
            "W": torch.tensor([[1.0, -1.0], [0.0, 2.0]], device=dev, dtype=f32),
            "bias": torch.tensor([0.0, 1.0], device=dev, dtype=f32),
            "y": torch.zeros(2, 2, device=dev, dtype=f32),
            "M": 2,
            "K": 2,
            "N": 2,
            "eps": 1e-5,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]

        # Minimal: one row, K=1 (normalized value is 0, output is GELU(beta*W + bias)).
        tests.append(self._make(1, 1, 3, seed=1))

        # Constant rows (zero variance): eps must be added inside the sqrt or result is nan.
        c = self._make(3, 4, 5, eps=1e-5, seed=2)
        c["x"].fill_(0.0)
        tests.append(c)

        # Large eps relative to variance: catches implementations that ignore eps.
        tests.append(self._make(4, 8, 4, eps=1.0, seed=3, x_range=(-0.5, 0.5)))

        # Negative-only inputs with a large offset: catches one-pass E[x^2]-E[x]^2 cancellation.
        tests.append(self._make(16, 64, 32, seed=4, x_range=(-1003.0, -997.0)))

        # Non-power-of-two dims (tile tails in M, N and K).
        tests.append(self._make(30, 100, 70, seed=5))

        # Power-of-two dims.
        tests.append(self._make(128, 256, 512, seed=6))

        # Non-square, N > K.
        tests.append(self._make(255, 64, 1000, seed=7))

        # Realistic transformer MLP up-projection.
        tests.append(self._make(512, 1024, 4096, seed=8))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make(4096, 4096, 4096, seed=0)
