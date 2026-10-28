import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "Fused RoPE Causal Attention"
    # Same float32 tolerance as Multi-Head Attention (#12) and Softcapped Sliding Window
    # Attention (#152). The rotation is orthogonal (cos^2 + sin^2 = 1 to float32 rounding), so it
    # does not change the size of Q or K, and the outputs are convex combinations of V rows, so
    # the error does not grow with the sequence length. On CPU the float32 reference stays well
    # inside atol + rtol * max|ref| of an independent float64 implementation on every functional
    # case and the performance case (worst on the 4,096-token performance case). Interleaved
    # (GPT-J style) pairs instead of rotate-half pairs, rotating by -theta, rotating only Q or
    # only K, rotating V, a missing causal mask or h % num_kv_heads head mapping all miss by
    # orders of magnitude more (see the targeted cases below).
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        output: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
    ):
        assert head_dim % 2 == 0
        half = head_dim // 2
        assert Q.shape == (num_q_heads, seq_len, head_dim)
        assert K.shape == (num_kv_heads, seq_len, head_dim)
        assert V.shape == (num_kv_heads, seq_len, head_dim)
        assert cos.shape == (seq_len, half)
        assert sin.shape == (seq_len, half)
        assert output.shape == (num_q_heads, seq_len, head_dim)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32
        assert cos.dtype == sin.dtype == torch.float32
        assert num_q_heads % num_kv_heads == 0

        group_size = num_q_heads // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        # Rotate-half (Llama / GPT-NeoX) RoPE: element d of the first half is paired with element
        # d + head_dim / 2, and the pair is rotated by the angle whose cos / sin are stored at
        # cos[pos, d] / sin[pos, d].
        def rope(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., :half]
            x2 = x[..., half:]
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

        q_rot = rope(Q)
        k_rot = rope(K)

        rows = torch.arange(seq_len, device=Q.device).unsqueeze(1)
        cols = torch.arange(seq_len, device=Q.device).unsqueeze(0)
        allowed = cols <= rows

        heads = []
        for h in range(num_q_heads):
            g = h // group_size
            scores = torch.matmul(q_rot[h], k_rot[g].t()) * scale
            scores = scores.masked_fill(~allowed, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(weights, V[g]))
        output.copy_(torch.stack(heads, dim=0))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "cos": (ctypes.POINTER(ctypes.c_float), "in"),
            "sin": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_q_heads": (ctypes.c_int, "in"),
            "num_kv_heads": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _tables(self, seq_len: int, head_dim: int, base: float = 10000.0):
        # Standard RoPE tables: theta[pos, d] = pos * base^(-2d / head_dim). The angles are
        # computed in float64 on the CPU and rounded once to float32, like a precomputed cache.
        half = head_dim // 2
        pos = torch.arange(seq_len, dtype=torch.float64).unsqueeze(1)
        inv_freq = base ** (-torch.arange(half, dtype=torch.float64) * 2.0 / head_dim)
        theta = pos * inv_freq.unsqueeze(0)
        cos = torch.cos(theta).to(torch.float32).to(self.device)
        sin = torch.sin(theta).to(torch.float32).to(self.device)
        return cos, sin

    def _rand(self, shape, low: float, high: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(low, high)

    def _case(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Dict[str, Any]:
        num_q_heads, seq_len, head_dim = Q.shape
        num_kv_heads = K.shape[0]
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "cos": cos,
            "sin": sin,
            "output": torch.full(
                (num_q_heads, seq_len, head_dim), 123.0, device=self.device, dtype=torch.float32
            ),
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def _rand_case(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        qk_range: float,
        seed: int,
        base: float = 10000.0,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        Q = self._rand((num_q_heads, seq_len, head_dim), -qk_range, qk_range)
        K = self._rand((num_kv_heads, seq_len, head_dim), -qk_range, qk_range)
        V = self._rand((num_kv_heads, seq_len, head_dim), -1.0, 1.0)
        cos, sin = self._tables(seq_len, head_dim, base)
        return self._case(Q, K, V, cos, sin)

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]],
            device=self.device,
            dtype=dtype,
        )
        K = torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]],
            device=self.device,
            dtype=dtype,
        )
        V = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            device=self.device,
            dtype=dtype,
        )
        # Pair 0 (dims 0 and 2) rotates by 90 degrees per position; pair 1 (dims 1 and 3) is not
        # rotated.
        cos = torch.tensor([[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]], device=self.device, dtype=dtype)
        sin = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], device=self.device, dtype=dtype)
        return self._case(Q, K, V, cos, sin)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests: List[Dict[str, Any]] = []

        # Every output buffer is pre-filled with 123.0, so a kernel that skips heads, rows or a
        # tile tail fails instead of passing on leftover zeros.
        #
        # 1: basic example (same as generate_example_test). Q and K rows are identical, so all
        # scores would be equal without RoPE; with RoPE the score depends on the relative angle
        # (0, 90 or 180 degrees on pair 0), which a kernel that skips the rotation misses.
        tests.append(self.generate_example_test())

        # 2: single token, two query heads sharing one KV head: the only visible key is the
        # token itself, so output[h, 0] = V[0, 0] for both heads whatever the rotation.
        cos, sin = self._tables(1, 4)
        tests.append(
            self._case(
                torch.tensor(
                    [[[1.0, -2.0, 0.5, 3.0]], [[-1.0, 0.0, 2.0, -0.5]]], device=self.device
                ),
                torch.tensor([[[0.5, 1.0, -1.0, 2.0]]], device=self.device),
                torch.tensor([[[3.0, -2.0, 1.0, 0.5]]], device=self.device),
                cos,
                sin,
            )
        )

        # 3: Q = K = 0: every visible key gets the same weight, so output[h, i] is the mean of
        # V[g, 0 .. i]. Catches a missing or off-by-one causal mask independent of the scores.
        torch.manual_seed(2)
        cos, sin = self._tables(7, 4)
        tests.append(
            self._case(
                torch.zeros((2, 7, 4), device=self.device, dtype=dtype),
                torch.zeros((1, 7, 4), device=self.device, dtype=dtype),
                self._rand((1, 7, 4), -2.0, 2.0),
                cos,
                sin,
            )
        )

        # 4: all-zero inputs (and all-zero tables)
        tests.append(
            self._case(
                torch.zeros((4, 16, 8), device=self.device, dtype=dtype),
                torch.zeros((2, 16, 8), device=self.device, dtype=dtype),
                torch.zeros((2, 16, 8), device=self.device, dtype=dtype),
                torch.zeros((16, 4), device=self.device, dtype=dtype),
                torch.zeros((16, 4), device=self.device, dtype=dtype),
            )
        )

        # 5: identity rotation (cos = 1, sin = 0): plain causal attention, checked against an
        # independent implementation. Catches errors in the attention itself (scale, softmax).
        torch.manual_seed(3)
        tests.append(
            self._case(
                self._rand((2, 32, 16), -2.0, 2.0),
                self._rand((2, 32, 16), -2.0, 2.0),
                self._rand((2, 32, 16), -1.0, 1.0),
                torch.ones((32, 8), device=self.device, dtype=dtype),
                torch.zeros((32, 8), device=self.device, dtype=dtype),
            )
        )

        # 6: random, non-uniform tables (unit-norm pairs at arbitrary angles, different per
        # frequency): pairing d with d + head_dim / 2 versus interleaved (2d, 2d + 1) pairs,
        # rotating by -theta, or reading the table at the wrong position all change the scores.
        # Large Q and K make the softmax sharp, so those mistakes move the output a lot.
        torch.manual_seed(4)
        ang = self._rand((64, 16), -math.pi, math.pi)
        tests.append(
            self._case(
                self._rand((2, 64, 32), -3.0, 3.0),
                self._rand((1, 64, 32), -3.0, 3.0),
                self._rand((1, 64, 32), -1.0, 1.0),
                torch.cos(ang),
                torch.sin(ang),
            )
        )

        # 7: GQA head mapping with distinct KV heads: query head h must read KV head
        # h // (num_q_heads / num_kv_heads). Mapping with h % num_kv_heads, or always using
        # KV head 0, gives different outputs for heads 1-6.
        tests.append(self._rand_case(8, 2, 64, 16, qk_range=2.0, seed=5))

        # 8: multi-head (num_kv_heads = num_q_heads), non-power-of-two sequence length and a
        # head_dim (34) whose half (17) is odd, so vectorized loads over pairs need tails.
        tests.append(self._rand_case(4, 4, 255, 34, qk_range=2.0, seed=6))

        # 9: multi-query (one KV head), tile tails in the sequence, head_dim = 128, base 500,000
        # (Llama 3 style) tables
        tests.append(self._rand_case(8, 1, 1000, 128, qk_range=1.0, seed=7, base=500000.0))

        # 10: realistic size: performance-test head layout and value range, shorter sequence
        torch.manual_seed(8)
        Q = torch.randn((16, 2048, 64), device=self.device, dtype=dtype)
        K = torch.randn((4, 2048, 64), device=self.device, dtype=dtype)
        V = torch.randn((4, 2048, 64), device=self.device, dtype=dtype)
        cos, sin = self._tables(2048, 64)
        tests.append(self._case(Q, K, V, cos, sin))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        num_q_heads, num_kv_heads, seq_len, head_dim = 16, 4, 4096, 64
        cos, sin = self._tables(seq_len, head_dim)
        return {
            "Q": RandnTensor((num_q_heads, seq_len, head_dim)),
            "K": RandnTensor((num_kv_heads, seq_len, head_dim)),
            "V": RandnTensor((num_kv_heads, seq_len, head_dim)),
            "cos": cos,
            "sin": sin,
            "output": OutTensor((num_q_heads, seq_len, head_dim)),
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }
