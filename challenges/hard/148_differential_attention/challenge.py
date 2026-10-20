import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Differential Attention"
    # Each head output is RMS-normalized, so outputs are O(1) (RMS = 1 - lambda_init). The
    # float64 reference keeps float32 / online-softmax solutions within ~1e-5. Dropping the
    # causal mask, attending one key too few or too many, pairing Q1 with K2, adding instead of
    # subtracting the second map, normalizing over head_dim instead of 2 * head_dim, skipping
    # the (1 - lambda_init) scale or using the wrong softmax scale are all off by far more
    # than 1e-3 on the targeted cases below.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    eps = 1e-5

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        lambdas: torch.Tensor,
        output: torch.Tensor,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        lambda_init: float,
    ):
        S, H, D = seq_len, num_heads, head_dim
        assert Q.shape == K.shape == V.shape == output.shape == (S, H, 2 * D)
        assert lambdas.shape == (H,)
        assert Q.dtype == K.dtype == V.dtype == lambdas.dtype == output.dtype == torch.float32

        scale = 1.0 / math.sqrt(D)
        causal = torch.ones(S, S, dtype=torch.bool, device=Q.device).tril()
        lam = lambdas.double()

        # One head at a time keeps the float64 score tensors at (S, S).
        for h in range(H):
            q = Q[:, h].double()
            k = K[:, h].double()
            v = V[:, h].double()
            s1 = (q[:, :D] @ k[:, :D].T) * scale
            s2 = (q[:, D:] @ k[:, D:].T) * scale
            a1 = torch.softmax(s1.masked_fill(~causal, float("-inf")), dim=-1)
            a2 = torch.softmax(s2.masked_fill(~causal, float("-inf")), dim=-1)
            o = (a1 - lam[h] * a2) @ v  # (S, 2D)
            o = o * torch.rsqrt(o.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            output[:, h] = (o * (1.0 - lambda_init)).float()

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "lambdas": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "seq_len": (ctypes.c_int, "in"),
            "num_heads": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
            "lambda_init": (ctypes.c_float, "in"),
        }

    def _make(
        self,
        S,
        H,
        D,
        seed=0,
        q_scale=1.0,
        lambdas=None,
        lambda_init=0.5,
        zero_q=False,
        zero_v=False,
    ):
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        if zero_q:
            Q = torch.zeros(S, H, 2 * D, device=dev, dtype=f32)
        else:
            Q = torch.randn(S, H, 2 * D, device=dev, dtype=f32) * q_scale
        K = torch.randn(S, H, 2 * D, device=dev, dtype=f32)
        if zero_v:
            V = torch.zeros(S, H, 2 * D, device=dev, dtype=f32)
        else:
            V = torch.randn(S, H, 2 * D, device=dev, dtype=f32)
        if lambdas is None:
            lam = torch.rand(H, device=dev, dtype=f32) * 0.6 + 0.2
        else:
            lam = torch.tensor(lambdas, device=dev, dtype=f32)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "lambdas": lam,
            "output": torch.full((S, H, 2 * D), 123.0, device=dev, dtype=f32),
            "seq_len": S,
            "num_heads": H,
            "head_dim": D,
            "lambda_init": lambda_init,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        # S=2, H=1, D=1, lambda=0.5, lambda_init=0.2. Q2 = 0, so the second map is uniform.
        # Row 0 sees key 0 only: diff = (1 - 0.5) * V0 = [1, 0] -> RMS-normalized [1.4142, 0]
        # -> * 0.8 = [1.1314, 0].
        # Row 1: a1 = softmax([0, 1]) = [0.2689, 0.7311], a2 = [0.5, 0.5],
        # a1 - 0.5 * a2 = [0.0189, 0.4811], diff = [0.0379, 0.9621] -> [0.0445, 1.1305].
        dev, f32 = self.device, torch.float32
        return {
            "Q": torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], device=dev, dtype=f32),
            "K": torch.tensor([[[0.0, 5.0]], [[1.0, 7.0]]], device=dev, dtype=f32),
            "V": torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]], device=dev, dtype=f32),
            "lambdas": torch.tensor([0.5], device=dev, dtype=f32),
            "output": torch.full((2, 1, 2), 123.0, device=dev, dtype=f32),
            "seq_len": 2,
            "num_heads": 1,
            "head_dim": 1,
            "lambda_init": 0.2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]

        # Single token: both maps are [1], so output = (1 - lambda_init) * V / rms(V) with the
        # sign of (1 - lambda). lambda = -0.5 catches clamping lambda to [0, 1].
        tests.append(self._make(1, 1, 1, seed=1, lambdas=[-0.5], lambda_init=0.3))

        # Zero queries: both maps are uniform causal averages, diff = (1 - lambda) * mean(V).
        # Catches applying lambda to the first map or adding the second map.
        tests.append(self._make(4, 2, 4, seed=2, zero_q=True, lambdas=[0.25, 0.75]))

        # Zero values: the difference is exactly 0, so the RMS norm must use eps and write 0,
        # not NaN, and must overwrite the pre-filled output.
        tests.append(self._make(3, 2, 2, seed=3, zero_v=True))

        # Tiny values (|O| ~ 1e-3, so mean(O^2) is comparable to eps): catches dropping eps
        # or adding it outside the square root.
        t = self._make(6, 2, 4, seed=13, lambdas=[0.3, 0.6])
        t["V"] *= 1e-3
        tests.append(t)

        # Sharp softmax (scores up to ~+-30) with negative inputs on a tiny sequence: attending
        # one key past the diagonal, or one key short of it, is far off.
        tests.append(self._make(3, 2, 2, seed=4, q_scale=6.0, lambdas=[0.8, 0.1]))

        # Identical maps (Q1 = Q2, K1 = K2) with lambda = 1: diff is exactly 0 for every row,
        # so the output must be 0. Catches adding the second map and an even/odd feature split.
        t = self._make(5, 2, 4, seed=5, q_scale=2.0, lambdas=[1.0, 1.0], lambda_init=0.6)
        t["Q"][:, :, 4:] = t["Q"][:, :, :4]
        t["K"][:, :, 4:] = t["K"][:, :, :4]
        tests.append(t)

        # Q1 and K2 scaled very differently from Q2 and K1: a Q1/K2 pairing or an interleaved
        # (even/odd) split of the 2 * head_dim features gives a different map.
        t = self._make(16, 4, 8, seed=6, q_scale=2.0, lambdas=[0.2, 0.4, 0.6, 0.9])
        t["K"][:, :, 8:] *= 0.1
        tests.append(t)

        # Per-head lambdas spread out (negative, zero and one), large lambda_init: catches
        # using lambdas[0] for every head and skipping the (1 - lambda_init) scale.
        tests.append(
            self._make(
                32,
                8,
                16,
                seed=7,
                lambdas=[-1.0, -0.3, 0.0, 0.2, 0.45, 0.7, 0.9, 1.0],
                lambda_init=0.8,
            )
        )

        # Layout trap: seq_len != num_heads, non-power-of-2 sizes. Reading Q as
        # (H, S, 2D) or normalizing over head_dim instead of 2 * head_dim fails.
        tests.append(self._make(30, 3, 8, seed=8, q_scale=1.5, lambda_init=0.1))
        tests.append(self._make(100, 5, 32, seed=9, q_scale=2.0))

        # Tile tails on realistic head sizes.
        tests.append(self._make(255, 4, 64, seed=10, lambda_init=0.35))
        tests.append(self._make(512, 8, 64, seed=11, q_scale=1.5, lambda_init=0.7))
        tests.append(self._make(1024, 2, 128, seed=12, lambda_init=0.2))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # One 32-head differential attention layer over a 4,096-token prompt.
        # Q + K + V + out = 4 * 4096 * 32 * 128 * 4 B ≈ 268 MB; 5x fits a T4's 16 GB.
        return self._make(4096, 32, 64, seed=0, lambda_init=0.8)
