import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "RMS Norm Backward"
    # dx is O(|weight| * |dy|) and dweight is a sum over up to N rows (O(sqrt(N)) for random data).
    # float32 error from a different reduction order is ~1e-6 relative. Dropping the projection
    # term, the weight in dx, rstd in dweight, or eps shifts results by >= 1e-2 on targeted cases.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
        dweight: torch.Tensor,
        N: int,
        C: int,
        eps: float,
    ):
        assert x.shape == (N, C)
        assert weight.shape == (C,)
        assert dy.shape == (N, C)
        assert dx.shape == (N, C)
        assert dweight.shape == (C,)
        assert x.dtype == weight.dtype == dy.dtype == torch.float32
        assert dx.dtype == dweight.dtype == torch.float32

        rstd = torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + eps)
        xhat = x * rstd
        g = dy * weight
        proj = torch.mean(g * xhat, dim=1, keepdim=True)
        dx.copy_(rstd * (g - xhat * proj))
        dweight.copy_(torch.sum(dy * xhat, dim=0))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "dy": (ctypes.POINTER(ctypes.c_float), "in"),
            "dx": (ctypes.POINTER(ctypes.c_float), "out"),
            "dweight": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def _outputs(self, N, C):
        # Outputs are pre-filled with garbage so solutions that accumulate into dweight
        # without initializing it (e.g. atomicAdd onto an assumed-zero buffer) fail.
        dev, f32 = self.device, torch.float32
        return {
            "dx": torch.full((N, C), 123.0, device=dev, dtype=f32),
            "dweight": torch.full((C,), 123.0, device=dev, dtype=f32),
        }

    def _make(
        self,
        N,
        C,
        eps=1e-5,
        seed=0,
        x_scale=1.0,
        w_range=(0.5, 1.5),
        dy_scale=1.0,
        correlated=False,
    ):
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        x = torch.randn(N, C, device=dev, dtype=f32) * x_scale
        weight = torch.empty(C, device=dev, dtype=f32).uniform_(*w_range)
        dy = torch.randn(N, C, device=dev, dtype=f32) * dy_scale
        if correlated:
            # dy aligned with x makes mean(g * xhat) O(1) instead of O(1/sqrt(C)),
            # so a missing projection term produces a large error.
            dy = dy * 0.25 + x / x_scale
        return {
            "x": x,
            "weight": weight,
            "dy": dy,
            **self._outputs(N, C),
            "N": N,
            "C": C,
            "eps": eps,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "x": torch.tensor(
                [[1.0, -1.0, 1.0, -1.0], [2.0, 0.0, 0.0, 0.0]], device=dev, dtype=f32
            ),
            "weight": torch.tensor([1.0, 2.0, 0.5, 1.0], device=dev, dtype=f32),
            "dy": torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]], device=dev, dtype=f32),
            **self._outputs(2, 4),
            "N": 2,
            "C": 4,
            "eps": 1e-5,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Hand-computed example (eps ~ 0): dx = [[1.375, 1.625, 0.875, 0.625], [0, 0, 0, 1]],
        # dweight = [3, -1, 1, -1]. Checks the reference against independent values.
        tests.append(self.generate_example_test())

        # N = 1, C = 1: xhat ~ sign(x) and dx ~ 0 (the projection cancels the whole gradient);
        # catches implementations that drop the projection term.
        tests.append(self._make(1, 1, seed=1))

        # Single row: dweight = dy * xhat for that row only (no cross-row reduction).
        tests.append(self._make(1, 3, seed=2))

        # All-zero x: rstd = 1/sqrt(eps) is finite only if eps is inside the sqrt;
        # dweight is exactly 0 and dx = dy * weight / sqrt(eps).
        z = self._make(3, 4, seed=3)
        z["x"].zero_()
        tests.append(z)

        # Large eps relative to mean(x^2): catches implementations that ignore eps.
        tests.append(self._make(4, 8, eps=1.0, seed=4, x_scale=0.3))

        # Zero upstream gradient: both outputs must be exactly 0 (catches unset outputs).
        zg = self._make(5, 16, seed=5)
        zg["dy"].zero_()
        tests.append(zg)

        # Negative weights and dy correlated with x: large projection term; catches using dy
        # instead of dy * weight in dx, and a missing or wrongly-scaled projection.
        tests.append(self._make(16, 64, seed=6, w_range=(-2.0, -0.5), correlated=True))

        # Large-magnitude x (|x| ~ 50): rstd << 1, catches using x instead of xhat in dweight
        # and forgetting the outer rstd factor in dx.
        tests.append(self._make(32, 128, seed=7, x_scale=50.0, w_range=(-1.0, 1.0)))

        # Non-power-of-two, non-square shapes with N != C: catches row/column mix-ups,
        # dividing by N instead of C, and column-tile tails.
        tests.append(self._make(100, 30, seed=8, correlated=True))
        tests.append(self._make(255, 1000, seed=9, w_range=(-1.5, 1.5)))

        # Many rows, odd count: exercises multi-block dweight reduction with a ragged last
        # row-chunk; catches overwriting instead of accumulating partial sums across blocks.
        tests.append(self._make(4099, 256, seed=10, correlated=True))

        # Realistic LLM shape (hidden size 2048).
        tests.append(self._make(1024, 2048, seed=11, w_range=(-1.0, 1.0)))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # 16,384 tokens (e.g. batch 4 x seq 4,096) with LLaMA-7B hidden size 4,096.
        return self._make(16384, 4096, seed=0, correlated=True)
