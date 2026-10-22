import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused RMSNorm and W8A8 Linear"
    # Outputs are O(1). A float32 solution can round a handful of activations whose scaled
    # value lands within ~1e-5 of a .5 boundary to the neighbouring int8 level; each such
    # flip moves the affected outputs by at most x_scale * w_scale * 127 (< 4e-3 except on the
    # small-K outlier cases, where flips are rare), so 1e-2 absorbs them. Skipping activation
    # quantization, using one scale for the whole tensor, truncating instead of rounding,
    # dropping RMSNorm, gamma, eps, w_scale or the per-token scale, or wrapping the
    # accumulator in 16 bits are all far outside 1e-2 on the targeted cases below.
    atol = 1e-02
    rtol = 1e-02
    num_gpus = 1
    access_tier = "free"

    eps = 1e-6

    def reference_impl(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        W: torch.Tensor,
        w_scale: torch.Tensor,
        output: torch.Tensor,
        M: int,
        N: int,
        K: int,
    ):
        assert x.shape == (M, K) and gamma.shape == (K,)
        assert W.shape == (N, K) and w_scale.shape == (N,) and output.shape == (M, N)
        assert x.dtype == gamma.dtype == w_scale.dtype == output.dtype == torch.float32
        assert W.dtype == torch.int8

        # RMSNorm and per-token symmetric int8 quantization in float64.
        xd = x.double()
        y = xd * torch.rsqrt(xd.pow(2).mean(dim=1, keepdim=True) + self.eps) * gamma.double()
        x_scale = y.abs().amax(dim=1, keepdim=True) / 127.0  # (M, 1)
        safe = torch.where(x_scale > 0, x_scale, torch.ones_like(x_scale))
        q = torch.round(y / safe).clamp(-127, 127)
        q = torch.where(x_scale > 0, q, torch.zeros_like(q))

        # Exact integer accumulation: every product is an integer with |q * w| <= 127 * 127,
        # so float32 sums over chunks of 1,024 stay below 2^24 and are exact.
        qf = q.float()
        wf = W.float()
        acc = torch.zeros(M, N, dtype=torch.float64, device=x.device)
        for k0 in range(0, K, 1024):
            acc += (qf[:, k0 : k0 + 1024] @ wf[:, k0 : k0 + 1024].T).double()

        output.copy_((acc * x_scale * w_scale.double()).float())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "W": (ctypes.POINTER(ctypes.c_int8), "in"),
            "w_scale": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def _make(
        self,
        M,
        N,
        K,
        seed=0,
        x_scale=1.0,
        outliers=0,
        outlier_mag=40.0,
        positive=False,
        zero_rows=(),
    ):
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        x = torch.randn(M, K, device=dev, dtype=f32)
        if positive:
            x = x.abs() + 0.5
        if outliers:
            # A few fixed outlier channels, as in real LLM activations.
            idx = torch.randperm(K, device=dev)[:outliers]
            x[:, idx] *= outlier_mag
        x *= x_scale
        for r in zero_rows:
            x[r] = 0.0
        gamma = torch.rand(K, device=dev, dtype=f32) + 0.5
        if positive:
            W = torch.randint(1, 128, (N, K), device=dev, dtype=torch.int8)
        else:
            W = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        # Real weights are W * w_scale, about U(-1, 1) / sqrt(K), so outputs are O(1).
        w_scale = (torch.rand(N, device=dev, dtype=f32) + 0.5) / (127.0 * K**0.5)
        return {
            "x": x,
            "gamma": gamma,
            "W": W,
            "w_scale": w_scale,
            "output": torch.full((M, N), 123.0, device=dev, dtype=f32),
            "M": M,
            "N": N,
            "K": K,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        # Row 0 = [3, -1, 0, 1]: mean(x^2) = 2.75, y = x / sqrt(2.75 + 1e-6) (gamma = 1),
        # max |y| = 1.80907, x_scale = 0.0142446, q = round(127 * x / 3) = [127, -42, 0, 42].
        # Row 1 = [1, 1, 1, 1]: y ~ 1, x_scale ~ 1 / 127, q = [127, 127, 127, 127].
        # output[0] = 0.0142446 * [127 * 0.5, -84 * 0.25, 1270 * 0.01] = [0.9045, -0.2991, 0.1809]
        # output[1] = (1 / 127) * [127 * 0.5, 0, 5080 * 0.01] = [0.5, 0, 0.4]
        dev, f32 = self.device, torch.float32
        return {
            "x": torch.tensor([[3.0, -1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]], device=dev, dtype=f32),
            "gamma": torch.ones(4, device=dev, dtype=f32),
            "W": torch.tensor(
                [[1, 0, 0, 0], [0, 2, -2, 0], [10, 10, 10, 10]], device=dev, dtype=torch.int8
            ),
            "w_scale": torch.tensor([0.5, 0.25, 0.01], device=dev, dtype=f32),
            "output": torch.full((2, 3), 123.0, device=dev, dtype=f32),
            "M": 2,
            "N": 3,
            "K": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]

        # Single element: y = x / sqrt(x^2 + eps) * gamma, always quantized to +-127.
        tests.append(self._make(1, 1, 1, seed=1, x_scale=3.0))

        # Tiny activations (mean(x^2) ~ eps): catches dropping eps or adding it outside the
        # square root.
        tests.append(self._make(3, 5, 8, seed=2, x_scale=1e-3))

        # Large activations and N != K: catches skipping RMSNorm (output scales with rms(x)),
        # reading W as (K, N), and indexing w_scale by row.
        tests.append(self._make(4, 7, 16, seed=3, x_scale=5.0))

        # All-zero token rows next to normal ones: the scale is 0 and the row must be 0,
        # not NaN, and must overwrite the pre-filled output.
        tests.append(self._make(5, 6, 12, seed=4, zero_rows=(0, 3)))

        # Outlier channels (the SmoothQuant case): ordinary features quantize to a few
        # levels, so skipping activation quantization, truncating, or a per-tensor scale are
        # all far off.
        tests.append(self._make(8, 16, 32, seed=5, outliers=1, outlier_mag=60.0))
        tests.append(self._make(16, 24, 64, seed=6, outliers=3, outlier_mag=20.0))

        # One outlier token among normal ones: a per-tensor scale crushes the normal rows.
        t = self._make(12, 20, 48, seed=7)
        t["x"][5] *= 1e3
        t["x"][5, :4] *= 50.0
        tests.append(t)

        # All-positive activations and weights: rounding errors cannot cancel, so floor or
        # truncation instead of round-to-nearest shows up as a bias. Also overflows a 16-bit
        # accumulator.
        tests.append(self._make(6, 10, 96, seed=8, positive=True))

        # Non-power-of-2 sizes and tile tails.
        tests.append(self._make(33, 65, 129, seed=9, x_scale=0.3))
        tests.append(self._make(100, 257, 384, seed=10, x_scale=0.7))
        tests.append(self._make(255, 300, 1000, seed=11, x_scale=2.0))

        # Multiple K chunks and realistic hidden sizes.
        tests.append(self._make(64, 512, 2048, seed=12, outliers=8, outlier_mag=10.0))
        tests.append(self._make(17, 1024, 4096, seed=13))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # A 4,096-token prefill through a 4,096 x 4,096 projection (one Llama-7B attention
        # projection). x + output = 2 * 4096 * 4096 * 4 B = 134 MB, W = 16.8 MB.
        return self._make(4096, 4096, 4096, seed=0, outliers=16, outlier_mag=10.0)
