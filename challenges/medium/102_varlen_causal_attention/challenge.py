import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Variable-Length Causal Attention"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output: torch.Tensor,
        T: int,
        d: int,
        S: int,
    ):
        assert Q.shape == K.shape == V.shape == output.shape == (T, d)
        assert cu_seqlens.shape == (S + 1,)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32
        assert cu_seqlens.dtype == torch.int32

        cu = cu_seqlens.to(torch.int64)
        seq_lens = cu[1:] - cu[:-1]
        seq_ids = torch.repeat_interleave(torch.arange(S, device=Q.device), seq_lens)
        positions = torch.arange(T, device=Q.device) - cu[:-1].repeat_interleave(seq_lens)

        scale = math.sqrt(d)
        scores = (Q @ K.T) / scale

        same_seq = seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0)
        causal = positions.unsqueeze(1) >= positions.unsqueeze(0)
        mask = same_seq & causal

        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        output.copy_(attn @ V)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "cu_seqlens": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "T": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
        }

    def _pack_from_lengths(self, lengths: List[int], d: int):
        S = len(lengths)
        T = sum(lengths)
        device = self.device
        cu = torch.zeros(S + 1, device=device, dtype=torch.int32)
        cu[1:] = torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0)
        Q = torch.randn(T, d, device=device, dtype=torch.float32)
        K = torch.randn(T, d, device=device, dtype=torch.float32)
        V = torch.randn(T, d, device=device, dtype=torch.float32)
        output = torch.empty(T, d, device=device, dtype=torch.float32)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "cu_seqlens": cu,
            "output": output,
            "T": T,
            "d": d,
            "S": S,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        # Two sequences: lengths 2 and 1, total T=3, d=4.
        # cu_seqlens = [0, 2, 3].
        # Identity-like Q/K so attention weights are easy to reason about.
        Q = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
        K = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
        V = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            device=device,
            dtype=dtype,
        )
        cu_seqlens = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
        output = torch.empty(3, 4, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "cu_seqlens": cu_seqlens,
            "output": output,
            "T": 3,
            "d": 4,
            "S": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(0)
        tests = []

        # Edge: single sequence of length 1 (only attends to itself).
        tests.append(self._pack_from_lengths([1], d=4))

        # Edge: a single sequence, behaves like standard causal attention.
        tests.append(self._pack_from_lengths([8], d=16))

        # Edge: many length-1 sequences — each query only sees its own key.
        tests.append(self._pack_from_lengths([1] * 16, d=8))

        # Mixed short lengths.
        tests.append(self._pack_from_lengths([3, 5, 2, 4], d=16))

        # Power-of-2 head dim, varied sequence lengths.
        tests.append(self._pack_from_lengths([16, 32, 8, 64], d=32))

        # Non-power-of-2 sequence lengths.
        tests.append(self._pack_from_lengths([7, 13, 5, 30, 11], d=64))

        # Zero inputs: all attention weights are uniform within sequence; output is zero.
        case = self._pack_from_lengths([6, 10, 4], d=16)
        case["Q"].zero_()
        case["K"].zero_()
        case["V"].zero_()
        tests.append(case)

        # Negative inputs.
        case = self._pack_from_lengths([12, 20], d=32)
        case["Q"].neg_()
        case["K"].neg_()
        case["V"].neg_()
        tests.append(case)

        # Realistic-ish: medium number of sequences with realistic lengths.
        tests.append(self._pack_from_lengths([128, 64, 192, 96, 32, 160, 48, 224], d=64))

        # Many short sequences with one long sequence.
        tests.append(self._pack_from_lengths([2, 2, 2, 2, 2, 2, 2, 2, 200], d=64))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(1)
        # 32 sequences, average length 256, total T=8192, head dim 64.
        # A naive (T,T) score matrix would be ~256 MB; an efficient solution
        # exploits the cu_seqlens structure to avoid that.
        lengths = [128, 256, 384, 192, 320, 256, 224, 288] * 4
        return self._pack_from_lengths(lengths, d=64)
