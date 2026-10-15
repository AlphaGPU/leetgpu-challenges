import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Flash Decoding Attention"
    # Softmax-weighted averages of O(1) values over up to 65K keys; float64
    # reference keeps error near 1e-6 in float32, while attending to the wrong
    # keys, dropping 1/sqrt(D) or using an unstable softmax is off by >> 1e-3.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        T: int,
        S: int,
        H: int,
        D: int,
    ):
        assert Q.shape == (T, H, D)
        assert K.shape == (S, H, D)
        assert V.shape == (S, H, D)
        assert output.shape == (T, H, D)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32

        # Accumulate attention in float64 for a stable reference against which
        # float32 / online-softmax implementations are checked.
        scale = 1.0 / math.sqrt(D)
        scores = torch.einsum("thd,shd->ths", Q.double(), K.double()) * scale
        attn = torch.softmax(scores, dim=-1)
        output.copy_(torch.einsum("ths,shd->thd", attn, V.double()).float())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "T": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def _make(self, T, S, H, D, seed=0, scale=1.0, zero_q=False):
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32

        def rnd(*shape):
            return torch.empty(*shape, device=dev, dtype=f32).uniform_(-scale, scale)

        Q = torch.zeros(T, H, D, device=dev, dtype=f32) if zero_q else rnd(T, H, D)
        return {
            "Q": Q,
            "K": rnd(S, H, D),
            "V": rnd(S, H, D),
            "output": torch.zeros(T, H, D, device=dev, dtype=f32),
            "T": T,
            "S": S,
            "H": H,
            "D": D,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "Q": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], device=dev, dtype=f32),
            "K": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], device=dev, dtype=f32),
            "V": torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]], device=dev, dtype=f32),
            "output": torch.zeros(2, 1, 2, device=dev, dtype=f32),
            "T": 2,
            "S": 3,
            "H": 1,
            "D": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]
        # Single query, single key: output equals V.
        tests.append(self._make(1, 1, 1, 1, seed=1))
        # Zero queries -> uniform attention over keys.
        tests.append(self._make(2, 4, 2, 4, seed=2, zero_q=True))
        # Negative-only values.
        neg = self._make(2, 8, 2, 8, seed=3)
        neg["V"].abs_().neg_()
        tests.append(neg)
        # Large scores: softmax without max-subtraction overflows.
        tests.append(self._make(4, 16, 2, 16, seed=4, scale=6.0))
        # Non-power-of-two shapes (decode batch of a few tokens).
        tests.append(self._make(3, 100, 3, 24, seed=5))
        # Power-of-two decode over a medium cache.
        tests.append(self._make(1, 1024, 8, 64, seed=6))
        tests.append(self._make(4, 2048, 4, 128, seed=7))
        # Multi-query decode over a long context (flash-decoding territory).
        tests.append(self._make(1, 8192, 16, 128, seed=8))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Long-context single-token decode: the flash-decoding regime.
        return self._make(1, 65536, 32, 128, seed=0)
