import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused Residual Add and LayerNorm with Saved Statistics"
    # y is O(|gamma| * 4 + |beta|). A tree/Welford reduction in float32 is accurate to ~1e-6
    # relative; the one-pass E[h^2] - E[h]^2 variance loses ~1e-2 on the large-offset tests,
    # and dropping eps, beta, or the residual add is off by far more than 1e-3.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        y: torch.Tensor,
        residual_out: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
        N: int,
        C: int,
        eps: float,
    ):
        assert x.shape == residual.shape == y.shape == residual_out.shape == (N, C)
        assert gamma.shape == beta.shape == (C,)
        assert mean.shape == rstd.shape == (N,)
        for t in (x, residual, gamma, beta, y, residual_out, mean, rstd):
            assert t.dtype == torch.float32

        # The sum is rounded to float32 exactly as the solution must store it; statistics
        # are computed in float64 from that rounded sum for a stable reference.
        h = x + residual
        hd = h.double()
        mu = hd.mean(dim=1)
        var = ((hd - mu[:, None]) ** 2).mean(dim=1)
        r = torch.rsqrt(var + eps)
        out = (hd - mu[:, None]) * r[:, None] * gamma.double() + beta.double()
        residual_out.copy_(h)
        y.copy_(out.float())
        mean.copy_(mu.float())
        rstd.copy_(r.float())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "residual": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_float), "out"),
            "residual_out": (ctypes.POINTER(ctypes.c_float), "out"),
            "mean": (ctypes.POINTER(ctypes.c_float), "out"),
            "rstd": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def _outputs(self, N, C):
        # Pre-filled with garbage so solutions that skip any output (or any row of it) fail.
        dev, f32 = self.device, torch.float32
        return {
            "y": torch.full((N, C), 123.0, device=dev, dtype=f32),
            "residual_out": torch.full((N, C), 123.0, device=dev, dtype=f32),
            "mean": torch.full((N,), 123.0, device=dev, dtype=f32),
            "rstd": torch.full((N,), 123.0, device=dev, dtype=f32),
        }

    def _make(self, N, C, eps=1e-5, seed=0, scale=1.0, offset=0.0, g_range=(0.5, 1.5)):
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        x = torch.randn(N, C, device=dev, dtype=f32) * scale + offset
        residual = torch.randn(N, C, device=dev, dtype=f32) * scale
        gamma = torch.empty(C, device=dev, dtype=f32).uniform_(*g_range)
        beta = torch.empty(C, device=dev, dtype=f32).uniform_(-0.5, 0.5)
        return {
            "x": x,
            "residual": residual,
            "gamma": gamma,
            "beta": beta,
            **self._outputs(N, C),
            "N": N,
            "C": C,
            "eps": eps,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "x": torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 2.0, 2.0]], device=dev, dtype=f32),
            "residual": torch.tensor(
                [[1.0, 0.0, -1.0, 0.0], [1.0, 1.0, -1.0, -1.0]], device=dev, dtype=f32
            ),
            "gamma": torch.tensor([1.0, 2.0, 1.0, 0.5], device=dev, dtype=f32),
            "beta": torch.tensor([0.0, 0.0, 1.0, 0.0], device=dev, dtype=f32),
            **self._outputs(2, 4),
            "N": 2,
            "C": 4,
            "eps": 1e-5,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Hand-checkable example: residual_out = [[2,2,2,4],[1,1,1,1]].
        tests.append(self.generate_example_test())

        # C = 1: variance is 0, so y = beta and rstd = 1/sqrt(eps) (eps must be inside sqrt).
        tests.append(self._make(1, 1, seed=1))

        # Constant rows (x + residual identical across C): y = beta exactly.
        c = self._make(3, 8, seed=2)
        c["x"].fill_(2.5)
        c["residual"].fill_(-1.0)
        tests.append(c)

        # Large eps relative to the variance: catches implementations that ignore eps.
        tests.append(self._make(4, 16, eps=1.0, seed=3, scale=0.3))

        # Zero residual: plain LayerNorm, residual_out must equal x.
        zr = self._make(5, 32, seed=4)
        zr["residual"].zero_()
        tests.append(zr)

        # Negative gamma, non-power-of-two widths, ragged tails.
        tests.append(self._make(7, 30, seed=5, g_range=(-2.0, -0.5)))
        tests.append(self._make(33, 1000, seed=6, scale=3.0, g_range=(-1.5, 1.5)))

        # Large DC offset (mean ~500, std ~1.4): the one-pass E[h^2] - E[h]^2 variance in
        # float32 suffers catastrophic cancellation; requires two-pass or Welford.
        tests.append(self._make(64, 1024, seed=7, offset=500.0))
        tests.append(self._make(17, 3000, seed=8, offset=-300.0, g_range=(-1.0, 1.0)))

        # Large magnitude (|h| ~ 100) and a tall, odd row count.
        tests.append(self._make(4099, 256, seed=9, scale=70.0))

        # Realistic LLM shapes: GPT-2 XL (1600) and a 5120-wide model.
        tests.append(self._make(512, 1600, seed=10))
        tests.append(self._make(256, 5120, seed=11, offset=20.0))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # 16,384 tokens (e.g. batch 4 x seq 4,096) with hidden size 4,096.
        return self._make(16384, 4096, seed=0)
