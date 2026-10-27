import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "Softcapped Sliding Window Attention"
    # Same float32 tolerance as Multi-Head Attention (#12) and Gated Attention (#147). The cap
    # bounds every score to [-softcap, softcap] and outputs are convex combinations of V rows,
    # so the error does not grow with the score range. On CPU the float32 reference stays within
    # 6% of atol + rtol * max|ref| of a float64 implementation (17% elementwise), worst on the
    # 2,048- and 4,096-token cases. A missing cap, the cap applied before the 1/sqrt(head_dim)
    # scale, an inclusive window, a missing causal mask or h % num_kv_heads head mapping all miss
    # by far more (see the targeted cases below).
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        window_size: int,
        softcap: float,
    ):
        assert Q.shape == (num_q_heads, seq_len, head_dim)
        assert K.shape == (num_kv_heads, seq_len, head_dim)
        assert V.shape == (num_kv_heads, seq_len, head_dim)
        assert output.shape == (num_q_heads, seq_len, head_dim)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32
        assert num_q_heads % num_kv_heads == 0
        assert window_size >= 1
        assert softcap > 0

        group_size = num_q_heads // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        # Query i sees key j iff j <= i (causal) and i - j < window_size (sliding window), so
        # every query sees itself and at most window_size keys in total.
        rows = torch.arange(seq_len, device=Q.device).unsqueeze(1)
        cols = torch.arange(seq_len, device=Q.device).unsqueeze(0)
        allowed = (cols <= rows) & (rows - cols < window_size)

        heads = []
        for h in range(num_q_heads):
            g = h // group_size
            scores = torch.matmul(Q[h], K[g].t()) * scale
            scores = softcap * torch.tanh(scores / softcap)
            scores = scores.masked_fill(~allowed, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(weights, V[g]))
        output.copy_(torch.stack(heads, dim=0))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_q_heads": (ctypes.c_int, "in"),
            "num_kv_heads": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
            "window_size": (ctypes.c_int, "in"),
            "softcap": (ctypes.c_float, "in"),
        }

    def _rand(self, shape, low: float, high: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(low, high)

    def _case(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        window_size: int,
        softcap: float,
    ) -> Dict[str, Any]:
        num_q_heads, seq_len, head_dim = Q.shape
        num_kv_heads = K.shape[0]
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": torch.full(
                (num_q_heads, seq_len, head_dim), 123.0, device=self.device, dtype=torch.float32
            ),
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "window_size": window_size,
            "softcap": softcap,
        }

    def _rand_case(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        window_size: int,
        softcap: float,
        qk_range: float,
        seed: int,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        Q = self._rand((num_q_heads, seq_len, head_dim), -qk_range, qk_range)
        K = self._rand((num_kv_heads, seq_len, head_dim), -qk_range, qk_range)
        V = self._rand((num_kv_heads, seq_len, head_dim), -1.0, 1.0)
        return self._case(Q, K, V, window_size, softcap)

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
                [[-2.0, 0.0], [0.0, -2.0], [2.0, -2.0]],
            ],
            device=self.device,
            dtype=dtype,
        )
        K = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]], device=self.device, dtype=dtype)
        V = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], device=self.device, dtype=dtype)
        return self._case(Q, K, V, window_size=2, softcap=1.0)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests: List[Dict[str, Any]] = []

        # Every output buffer is pre-filled with 123.0, so a kernel that skips heads, rows or a
        # tile tail fails instead of passing on leftover zeros.
        #
        # 1: basic example (same as generate_example_test). softcap = 1 squashes the scaled
        # scores (up to 2.83) hard, so dropping the cap, or applying tanh before the
        # 1/sqrt(head_dim) scale, changes the output well beyond tolerance.
        tests.append(self.generate_example_test())

        # 2: single token, two query heads sharing one KV head: the only visible key is the
        # token itself, so output[h, 0] = V[0, 0] for both heads.
        tests.append(
            self._case(
                torch.tensor(
                    [[[1.0, -2.0, 0.5, 3.0]], [[-1.0, 0.0, 2.0, -0.5]]], device=self.device
                ),
                torch.tensor([[[0.5, 1.0, -1.0, 2.0]]], device=self.device),
                torch.tensor([[[3.0, -2.0, 1.0, 0.5]]], device=self.device),
                window_size=1,
                softcap=5.0,
            )
        )

        # 3: window_size = 1: each token sees only itself, so output[h, i] = V[h // 2, i]
        # exactly. Catches an inclusive window (i - j <= window_size) that also lets in the
        # previous token.
        tests.append(self._rand_case(4, 2, 5, 8, window_size=1, softcap=10.0, qk_range=2.0, seed=1))

        # 4: Q = K = 0: every visible key gets the same weight, so output[h, i] is the mean of
        # V[g, max(0, i - 2) .. i] (window_size = 3). Catches window off-by-one errors and a
        # missing causal mask, independent of the score computation.
        torch.manual_seed(2)
        tests.append(
            self._case(
                torch.zeros((2, 7, 4), device=self.device, dtype=dtype),
                torch.zeros((1, 7, 4), device=self.device, dtype=dtype),
                self._rand((1, 7, 4), -2.0, 2.0),
                window_size=3,
                softcap=30.0,
            )
        )

        # 5: all-zero inputs
        tests.append(
            self._case(
                torch.zeros((4, 16, 8), device=self.device, dtype=dtype),
                torch.zeros((2, 16, 8), device=self.device, dtype=dtype),
                torch.zeros((2, 16, 8), device=self.device, dtype=dtype),
                window_size=4,
                softcap=50.0,
            )
        )

        # 6: window_size >= seq_len: the window never binds, so this is plain causal softcapped
        # attention. Catches kernels that clamp the window start incorrectly.
        tests.append(
            self._rand_case(2, 2, 32, 16, window_size=100, softcap=20.0, qk_range=2.0, seed=3)
        )

        # 7: small softcap with large scores: the scaled scores reach about +-30 but are capped
        # to +-2, so a missing cap (or a cap applied to the unscaled dot product) produces
        # near one-hot weights instead of soft ones.
        tests.append(
            self._rand_case(2, 1, 64, 32, window_size=20, softcap=2.0, qk_range=4.0, seed=4)
        )

        # 8: GQA head mapping with distinct KV heads: query head h must read KV head
        # h // (num_q_heads / num_kv_heads). Mapping with h % num_kv_heads, or always using
        # KV head 0, gives different outputs for heads 1-6.
        tests.append(
            self._rand_case(8, 2, 64, 16, window_size=16, softcap=10.0, qk_range=2.0, seed=5)
        )

        # 9: multi-head (num_kv_heads = num_q_heads), non-power-of-two sequence length and a
        # window that is not a multiple of typical tile sizes
        tests.append(
            self._rand_case(4, 4, 255, 32, window_size=37, softcap=15.0, qk_range=2.0, seed=6)
        )

        # 10: multi-query (one KV head), tile tails in both the sequence and the window
        tests.append(
            self._rand_case(8, 1, 1000, 64, window_size=129, softcap=10.0, qk_range=2.0, seed=7)
        )

        # 11: realistic size: performance-test head layout and value range, shorter sequence
        torch.manual_seed(8)
        Q = torch.randn((16, 2048, 64), device=self.device, dtype=dtype) * 2.0
        K = torch.randn((4, 2048, 64), device=self.device, dtype=dtype) * 2.0
        V = torch.randn((4, 2048, 64), device=self.device, dtype=dtype)
        tests.append(self._case(Q, K, V, window_size=512, softcap=10.0))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        num_q_heads, num_kv_heads, seq_len, head_dim = 16, 4, 4096, 64
        return {
            "Q": RandnTensor((num_q_heads, seq_len, head_dim), 0.0, 2.0),
            "K": RandnTensor((num_kv_heads, seq_len, head_dim), 0.0, 2.0),
            "V": RandnTensor((num_kv_heads, seq_len, head_dim)),
            "output": OutTensor((num_q_heads, seq_len, head_dim)),
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "window_size": 1024,
            "softcap": 10.0,
        }
