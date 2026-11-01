import ctypes
import math
from typing import Any, Dict, List, Optional

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Lightning Indexer Sparse Attention"
    # The indexer scores are sums of small integer products (|score| <= 2^23), so they are exact in
    # int32 and in float32 under any summation order, and the top-k selection (ties broken by the
    # lower position) is fully determined. The output is a softmax-weighted average of
    # V rows in [-1, 1] over <= 2,048 selected positions; float32 accumulation-order differences
    # stay below 2% of the atol + rtol * max|ref| budget. Selecting a different set of positions,
    # missing the 1/sqrt(D) scale or using half-precision probabilities misses by far more.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_w: torch.Tensor,
        output: torch.Tensor,
        B: int,
        S: int,
        H: int,
        D: int,
        H_I: int,
        D_I: int,
        topk: int,
    ):
        assert q.shape == (B, H, D), f"Expected q.shape=({B}, {H}, {D}), got {q.shape}"
        assert k_cache.shape == (B, S, D), f"Expected k_cache.shape=({B}, {S}, {D})"
        assert v_cache.shape == (B, S, D), f"Expected v_cache.shape=({B}, {S}, {D})"
        assert idx_q.shape == (B, H_I, D_I), f"Expected idx_q.shape=({B}, {H_I}, {D_I})"
        assert idx_k.shape == (B, S, D_I), f"Expected idx_k.shape=({B}, {S}, {D_I})"
        assert idx_w.shape == (B, H_I), f"Expected idx_w.shape=({B}, {H_I})"
        assert output.shape == (B, H, D), f"Expected output.shape=({B}, {H}, {D})"
        assert q.dtype == k_cache.dtype == v_cache.dtype == output.dtype == torch.float32
        assert idx_q.dtype == idx_k.dtype == idx_w.dtype == torch.int8

        # Indexer: I[b, s] = sum_j w[b, j] * relu(idx_q[b, j] . idx_k[b, s]). Every partial sum is
        # an integer of magnitude <= 2^23, so float32 matmuls compute it exactly.
        dots = torch.matmul(idx_k.to(torch.float32), idx_q.to(torch.float32).transpose(1, 2))
        scores = torch.matmul(torch.relu(dots), idx_w.to(torch.float32).unsqueeze(2)).squeeze(2)

        # Top-k by score, ties broken by the lower position: encode both in one int64 key.
        n_sel = min(topk, S)
        pos = torch.arange(S, device=q.device, dtype=torch.int64)
        key = scores.to(torch.int64) * S + (S - 1 - pos).unsqueeze(0)
        sel = torch.topk(key, n_sel, dim=1).indices  # (B, n_sel)

        gather_idx = sel.unsqueeze(2).expand(B, n_sel, D)
        k_sel = torch.gather(k_cache, 1, gather_idx)
        v_sel = torch.gather(v_cache, 1, gather_idx)

        logits = torch.matmul(q, k_sel.transpose(1, 2)) * (1.0 / math.sqrt(D))
        probs = torch.softmax(logits, dim=2)
        output.copy_(torch.matmul(probs, v_sel))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "q": (ctypes.POINTER(ctypes.c_float), "in"),
            "k_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "v_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "idx_q": (ctypes.POINTER(ctypes.c_int8), "in"),
            "idx_k": (ctypes.POINTER(ctypes.c_int8), "in"),
            "idx_w": (ctypes.POINTER(ctypes.c_int8), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "H_I": (ctypes.c_int, "in"),
            "D_I": (ctypes.c_int, "in"),
            "topk": (ctypes.c_int, "in"),
        }

    def _make(
        self,
        B: int,
        S: int,
        H: int,
        D: int,
        H_I: int,
        D_I: int,
        topk: int,
        seed: int = 0,
        q_scale: float = 4.0,
        idx_range: int = 8,
        w_range: int = 16,
        w_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        dev, f32, i8 = self.device, torch.float32, torch.int8
        q = torch.empty(B, H, D, device=dev, dtype=f32).uniform_(-1.0, 1.0) * q_scale
        k_cache = torch.empty(B, S, D, device=dev, dtype=f32).uniform_(-1.0, 1.0)
        v_cache = torch.empty(B, S, D, device=dev, dtype=f32).uniform_(-1.0, 1.0)
        idx_q = torch.randint(-idx_range, idx_range + 1, (B, H_I, D_I), device=dev, dtype=i8)
        idx_k = torch.randint(-idx_range, idx_range + 1, (B, S, D_I), device=dev, dtype=i8)
        if w_values is None:
            idx_w = torch.randint(-w_range, w_range + 1, (B, H_I), device=dev, dtype=i8)
        else:
            idx_w = torch.tensor(w_values, device=dev, dtype=i8).reshape(B, H_I)
        return {
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "idx_q": idx_q,
            "idx_k": idx_k,
            "idx_w": idx_w,
            "output": torch.full((B, H, D), 123.0, device=dev, dtype=f32),
            "B": B,
            "S": S,
            "H": H,
            "D": D,
            "H_I": H_I,
            "D_I": D_I,
            "topk": topk,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32, i8 = self.device, torch.float32, torch.int8
        return {
            "q": torch.tensor([[[1.0, 0.0]]], device=dev, dtype=f32),
            "k_cache": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]], device=dev, dtype=f32
            ),
            "v_cache": torch.tensor(
                [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=dev, dtype=f32
            ),
            "idx_q": torch.tensor([[[1, 2]]], device=dev, dtype=i8),
            "idx_k": torch.tensor([[[2, 1], [0, 1], [-1, 0], [0, 1]]], device=dev, dtype=i8),
            "idx_w": torch.tensor([[3]], device=dev, dtype=i8),
            "output": torch.full((1, 1, 2), 123.0, device=dev, dtype=f32),
            "B": 1,
            "S": 4,
            "H": 1,
            "D": 2,
            "H_I": 1,
            "D_I": 2,
            "topk": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Example from the description: position 0 has the top score (12) and positions 1 and 3
        # tie at 6; the tie must go to position 1 (lower position wins). Hand-checked output.
        tests.append(self.generate_example_test())

        # Single cached token, topk > S: the only position is selected (output = its V row
        # for every head); catches solutions that read past S or pad the selection.
        tests.append(self._make(1, 1, 3, 8, 2, 4, topk=4, seed=1))

        # topk > S with several tokens: every position is attended exactly once (catches
        # duplicating position 0 or uninitialized indices to fill k slots).
        tests.append(self._make(2, 5, 2, 16, 4, 8, topk=8, seed=2))

        # All indexer keys are zero, so every score ties at 0: the selection must be positions
        # 0..topk-1 (catches breaking ties toward higher positions or unstable selection).
        tied = self._make(2, 64, 4, 16, 4, 16, topk=10, seed=3)
        tied["idx_k"].zero_()
        tests.append(tied)

        # Small integer range (-1..1) creates many ties straddling the top-k boundary.
        tests.append(self._make(2, 300, 4, 32, 4, 8, topk=37, seed=4, idx_range=1, w_range=2))

        # Mixed-sign indexer weights: relu must be applied to each head's dot product before
        # weighting (relu(w * dot) and weight-free or relu-free scoring select different sets).
        tests.append(self._make(1, 200, 4, 32, 4, 16, topk=20, seed=5, w_values=[5, -7, 3, -2]))

        # Batch of 3 with distinct caches: catches sharing batch 0's selection, or indexing the
        # cache of batch b with the wrong stride.
        tests.append(self._make(3, 128, 8, 32, 8, 32, topk=16, seed=6))

        # Zero queries: attention is the plain mean of the selected V rows.
        zq = self._make(1, 100, 4, 24, 4, 16, topk=25, seed=7)
        zq["q"].zero_()
        tests.append(zq)

        # Large attention logits (|q.k| / sqrt(D) in the hundreds): softmax without max-subtraction
        # overflows.
        tests.append(self._make(2, 256, 4, 64, 4, 32, topk=64, seed=8, q_scale=150.0))

        # Non-power-of-two everything.
        tests.append(self._make(3, 1000, 7, 40, 5, 24, topk=100, seed=9))

        # Power-of-two shapes.
        tests.append(self._make(2, 4096, 16, 64, 8, 64, topk=256, seed=10))

        # Realistic: long context, DeepSeek-style 2,048 selected positions, 64 indexer heads.
        tests.append(self._make(2, 16384, 32, 128, 64, 128, topk=2048, seed=11))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Decode step for 4 sequences with 65,536 cached tokens, 32 query heads sharing one
        # 128-dim KV head (MQA), a 64-head x 128-dim int8 indexer and 2,048 selected positions.
        return self._make(4, 65536, 32, 128, 64, 128, topk=2048, seed=157)
