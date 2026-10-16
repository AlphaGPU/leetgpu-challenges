import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Layer Norm Backward"
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
        dgamma: torch.Tensor,
        dbeta: torch.Tensor,
        N: int,
        C: int,
    ):
        assert x.shape == dy.shape == dx.shape == (N, C)
        assert gamma.shape == dgamma.shape == dbeta.shape == (C,)
        assert mean.shape == rstd.shape == (N,)
        assert (
            x.dtype
            == gamma.dtype
            == mean.dtype
            == rstd.dtype
            == dy.dtype
            == dx.dtype
            == dgamma.dtype
            == dbeta.dtype
            == torch.float32
        )

        xhat = (x - mean[:, None]) * rstd[:, None]
        dgamma.copy_(torch.sum(dy * xhat, dim=0))
        dbeta.copy_(torch.sum(dy, dim=0))
        g = dy * gamma
        proj = torch.mean(g * xhat, dim=1, keepdim=True)
        mean_g = torch.mean(g, dim=1, keepdim=True)
        dx.copy_(rstd[:, None] * (g - mean_g - xhat * proj))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "mean": (ctypes.POINTER(ctypes.c_float), "in"),
            "rstd": (ctypes.POINTER(ctypes.c_float), "in"),
            "dy": (ctypes.POINTER(ctypes.c_float), "in"),
            "dx": (ctypes.POINTER(ctypes.c_float), "out"),
            "dgamma": (ctypes.POINTER(ctypes.c_float), "out"),
            "dbeta": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
        }

    def _outputs(self, N, C):
        dev, f32 = self.device, torch.float32
        return {
            "dx": torch.full((N, C), 123.0, device=dev, dtype=f32),
            "dgamma": torch.full((C,), 123.0, device=dev, dtype=f32),
            "dbeta": torch.full((C,), 123.0, device=dev, dtype=f32),
        }

    def _make(
        self,
        N,
        C,
        eps=1e-5,
        seed=0,
        x_scale=1.0,
        gamma_range=(0.5, 1.5),
        dy_scale=1.0,
        correlated=False,
    ):
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        x = torch.randn(N, C, device=dev, dtype=f32) * x_scale
        gamma = torch.empty(C, device=dev, dtype=f32).uniform_(*gamma_range)
        dy = torch.randn(N, C, device=dev, dtype=f32) * dy_scale
        if correlated:
            dy = dy * 0.25 + x / x_scale

        mu = x.mean(dim=1)
        var = ((x - mu[:, None]) ** 2).mean(dim=1)
        rstd = torch.rsqrt(var + eps)

        return {
            "x": x,
            "gamma": gamma,
            "mean": mu,
            "rstd": rstd,
            "dy": dy,
            **self._outputs(N, C),
            "N": N,
            "C": C,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "x": torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 0.0, 1.0]], device=dev, dtype=f32),
            "gamma": torch.tensor([1.0, 1.0, 1.0, 1.0], device=dev, dtype=f32),
            "mean": torch.tensor([2.5, 0.0], device=dev, dtype=f32),
            "rstd": torch.tensor([0.89442719, 1.41421356], device=dev, dtype=f32),
            "dy": torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]], device=dev, dtype=f32),
            **self._outputs(2, 4),
            "N": 2,
            "C": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Hand-computed reference example
        tests.append(self.generate_example_test())

        # Single element edge case (N=1, C=1)
        tests.append(self._make(1, 1, seed=1))

        # Single row edge case (N=1, C=3)
        tests.append(self._make(1, 3, seed=2))

        # Uniform row with zero variance
        u = self._make(3, 4, seed=3)
        u["x"].fill_(2.5)
        u["mean"].fill_(2.5)
        u["rstd"].fill_(316.2277)
        tests.append(u)

        # Zero upstream gradient
        zg = self._make(5, 16, seed=4)
        zg["dy"].zero_()
        tests.append(zg)

        # Negative gamma with correlated dy
        tests.append(self._make(16, 64, seed=5, gamma_range=(-2.0, -0.5), correlated=True))

        # Large-magnitude inputs (|x| ~ 50)
        tests.append(self._make(32, 128, seed=6, x_scale=50.0, gamma_range=(-1.0, 1.0)))

        # Non-power-of-two non-square shapes
        tests.append(self._make(100, 30, seed=7, correlated=True))
        tests.append(self._make(255, 1000, seed=8, gamma_range=(-1.5, 1.5)))

        # Ragged row chunking (odd N)
        tests.append(self._make(4099, 256, seed=9, correlated=True))

        # Realistic hidden dimension (C=2048)
        tests.append(self._make(1024, 2048, seed=10, gamma_range=(-1.0, 1.0)))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make(16384, 4096, seed=0, correlated=True)
