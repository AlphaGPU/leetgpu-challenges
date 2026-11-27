import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "SnapKV Importance Scores"
    # Every score is a sum of at most num_groups * window <= 8,192 softmax probabilities, each
    # in [0, 1]. Against a float64 recomputation the float32 reference uses <= 0.4% of the
    # elementwise atol + rtol * |ref| budget on every case except the large-logit one (|scaled
    # logit| up to ~117), where it uses 12%; on the performance case it uses 0.2%. An independent
    # two-pass, key-tiled float32 implementation stays within 0.2% of the reference. Rounding Q
    # and K to TF32 / FP16 / BF16, dropping the 1/sqrt(d) scale, moving the causal boundary by
    # one, or normalizing with a stale running max all exceed the budget on functional cases.
    atol = 1e-05
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        scores: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        seq_len: int,
        head_dim: int,
    ):
        assert Q.shape == (num_q_heads, window, head_dim)
        assert K.shape == (num_kv_heads, seq_len, head_dim)
        assert scores.shape == (num_kv_heads, seq_len)
        assert Q.dtype == K.dtype == scores.dtype == torch.float32
        assert num_q_heads % num_kv_heads == 0
        assert 1 <= window <= seq_len

        num_groups = num_q_heads // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        # Query heads h = g * num_groups + j share KV head g: (num_kv_heads, groups * window, d)
        Qg = Q.reshape(num_kv_heads, num_groups * window, head_dim)
        logits = torch.bmm(Qg, K.transpose(1, 2)) * scale

        # Window query i sits at absolute position seq_len - window + i and sees keys t <= pos.
        q_pos = torch.arange(window, device=Q.device) + (seq_len - window)
        q_pos = q_pos.repeat(num_groups)
        k_pos = torch.arange(seq_len, device=Q.device)
        mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        scores.copy_(probs.sum(dim=1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "scores": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_q_heads": (ctypes.c_int, "in"),
            "num_kv_heads": (ctypes.c_int, "in"),
            "window": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _build(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        seq_len: int,
        head_dim: int,
        seed: int,
        std: float = 1.0,
        zero_q: bool = False,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        Q = torch.randn(num_q_heads, window, head_dim, device=dev, dtype=f32) * std
        K = torch.randn(num_kv_heads, seq_len, head_dim, device=dev, dtype=f32) * std
        if zero_q:
            Q.zero_()
        return {
            "Q": Q,
            "K": K,
            # Pre-filled with garbage: scores is not zero-initialized, so every key (including
            # tile tails) must be written and accumulating solutions must initialize it first.
            "scores": torch.full((num_kv_heads, seq_len), 123.0, device=dev, dtype=f32),
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "window": window,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        Q = torch.tensor(
            [
                [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 2.0]],
            ],
            device=dev,
            dtype=f32,
        )
        K = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]],
            device=dev,
            dtype=f32,
        )
        return {
            "Q": Q,
            "K": K,
            "scores": torch.full((1, 3), 123.0, device=dev, dtype=f32),
            "num_q_heads": 2,
            "num_kv_heads": 1,
            "window": 2,
            "seq_len": 3,
            "head_dim": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Hand-computed example: scores = [1.5495, 1.4516, 0.9989] (see challenge.html). Window
        # query 0 sits at position 1 and must not see key 2.
        tests.append(self.generate_example_test())

        # Minimal: one head, one key, one query. The only probability is 1, so scores = [1].
        tests.append(self._build(1, 1, 1, 1, 1, seed=1))

        # Zero queries: every visible logit is 0, so query i spreads 1 / (pos_i + 1) uniformly
        # over keys 0..pos_i. With seq_len = 5 and window = 3 the exact answer per KV head is
        # 2 * [1/3 + 1/4 + 1/5, 1/3 + 1/4 + 1/5, 1/3 + 1/4 + 1/5, 1/4 + 1/5, 1/5]. Catches a
        # missing causal mask (would give 2 * 3/5 everywhere), a mask that is off by one, and
        # queries placed at positions 0..window-1 instead of the end of the prompt.
        tests.append(self._build(4, 2, 3, 5, 8, seed=2, zero_q=True))

        # window == seq_len: the whole prompt is the observation window, so query 0 sees only
        # key 0. Catches end-alignment bugs at the boundary where the first query is at 0.
        tests.append(self._build(2, 1, 4, 4, 16, seed=3))

        # window = 1 (decode-like): no key is masked; catches masks that drop the diagonal.
        tests.append(self._build(4, 4, 1, 37, 32, seed=4))

        # Distinct KV heads with groups of 4: catches mapping query head h to KV head h % H_kv
        # instead of h // groups, and summing over only one head of the group.
        tests.append(self._build(8, 2, 8, 64, 32, seed=5))

        # Large logits (|scaled logit| up to ~117): exp overflows in float32 without max
        # subtraction; also catches a single-pass column accumulation that normalizes with a
        # stale running max instead of the row's final max and sum.
        tests.append(self._build(4, 1, 16, 300, 64, seed=6, std=5.0))

        # Non-power-of-2 everything: tile tails along keys, window and head_dim (d = 40).
        tests.append(self._build(6, 3, 30, 255, 40, seed=7))

        # Power-of-2 shapes with MHA (groups = 1) and head_dim 128.
        tests.append(self._build(4, 4, 32, 512, 128, seed=8))

        # Multi-query attention (one KV head, 16 query heads): long reduction over 16 * 24
        # queries per key.
        tests.append(self._build(16, 1, 24, 1000, 64, seed=9))

        # Realistic: 8 query heads, 2 KV heads, 32-token window over a 4,096-token prompt.
        tests.append(self._build(8, 2, 32, 4096, 64, seed=10))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Llama-3-8B-style GQA layer (32 query heads, 8 KV heads, head_dim 128) compressing a
        # 32K-token prompt with a 64-token observation window.
        num_q_heads, num_kv_heads, window, seq_len, head_dim = 32, 8, 64, 32768, 128
        return {
            "Q": RandnTensor((num_q_heads, window, head_dim)),
            "K": RandnTensor((num_kv_heads, seq_len, head_dim)),
            "scores": OutTensor((num_kv_heads, seq_len)),
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "window": window,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }
