import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "GQA Causal Attention Backward"
    # Tolerance: float32 with sums over up to seq_len keys (dQ) or group_size * seq_len queries
    # (dK, dV). On CPU the float32 reference and an independent float32 formulation (softmax
    # recomputed from scratch, D = rowsum(P * dP), K / V expanded per query head) both stay below
    # 5e-6 absolute error against a float64 autograd implementation on every functional case and
    # the performance case, where outputs are O(1)-O(10). A missing causal mask, a strict mask,
    # skipping the 1/sqrt(head_dim) scale, dropping the D term, the wrong query-to-KV head
    # mapping or keeping only one query head's contribution to dK / dV all miss by >= 1e-2.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        O: torch.Tensor,
        dO: torch.Tensor,
        lse: torch.Tensor,
        dQ: torch.Tensor,
        dK: torch.Tensor,
        dV: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
    ):
        assert num_q_heads % num_kv_heads == 0
        assert Q.shape == (num_q_heads, seq_len, head_dim)
        assert K.shape == (num_kv_heads, seq_len, head_dim)
        assert V.shape == (num_kv_heads, seq_len, head_dim)
        assert O.shape == (num_q_heads, seq_len, head_dim)
        assert dO.shape == (num_q_heads, seq_len, head_dim)
        assert lse.shape == (num_q_heads, seq_len)
        assert dQ.shape == (num_q_heads, seq_len, head_dim)
        assert dK.shape == (num_kv_heads, seq_len, head_dim)
        assert dV.shape == (num_kv_heads, seq_len, head_dim)
        for t in (Q, K, V, O, dO, lse, dQ, dK, dV):
            assert t.dtype == torch.float32

        group = num_q_heads // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)
        rows = torch.arange(seq_len, device=Q.device).unsqueeze(1)
        cols = torch.arange(seq_len, device=Q.device).unsqueeze(0)
        allowed = cols <= rows

        # D_i = rowsum(dO_i * O_i) equals sum_j P_ij * dP_ij, the softmax-Jacobian correction
        delta = (dO * O).sum(dim=-1)

        for g in range(num_kv_heads):
            hs = slice(g * group, (g + 1) * group)
            q = Q[hs]
            do = dO[hs]
            scores = torch.matmul(q, K[g].t()) * scale
            scores = scores.masked_fill(~allowed, float("-inf"))
            p = torch.exp(scores - lse[hs].unsqueeze(-1))
            dp = torch.matmul(do, V[g].t())
            ds = p * (dp - delta[hs].unsqueeze(-1)) * scale
            dQ[hs] = torch.matmul(ds, K[g])
            # K / V head g is shared by query heads g * group .. (g + 1) * group - 1, so its
            # gradients are the sum of those heads' contributions
            dV[g] = torch.matmul(p.transpose(1, 2), do).sum(dim=0)
            dK[g] = torch.matmul(ds.transpose(1, 2), q).sum(dim=0)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "O": (ctypes.POINTER(ctypes.c_float), "in"),
            "dO": (ctypes.POINTER(ctypes.c_float), "in"),
            "lse": (ctypes.POINTER(ctypes.c_float), "in"),
            "dQ": (ctypes.POINTER(ctypes.c_float), "out"),
            "dK": (ctypes.POINTER(ctypes.c_float), "out"),
            "dV": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_q_heads": (ctypes.c_int, "in"),
            "num_kv_heads": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
        }

    def _case(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, dO: torch.Tensor
    ) -> Dict[str, Any]:
        # Builds a test from the forward inputs: O and lse are what a float32 causal GQA forward
        # pass saves for the backward pass. Output buffers are pre-filled with 123.0, so a kernel
        # that skips rows, heads or a tile tail, or accumulates into the buffer without zeroing
        # it first, fails instead of passing on leftover zeros.
        num_q_heads, seq_len, head_dim = Q.shape
        num_kv_heads = K.shape[0]
        group = num_q_heads // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)
        rows = torch.arange(seq_len, device=self.device).unsqueeze(1)
        cols = torch.arange(seq_len, device=self.device).unsqueeze(0)
        allowed = cols <= rows
        out = torch.empty_like(Q)
        lse = torch.empty((num_q_heads, seq_len), device=self.device, dtype=torch.float32)
        for g in range(num_kv_heads):
            hs = slice(g * group, (g + 1) * group)
            scores = torch.matmul(Q[hs], K[g].t()) * scale
            scores = scores.masked_fill(~allowed, float("-inf"))
            lse[hs] = torch.logsumexp(scores, dim=-1)
            out[hs] = torch.matmul(torch.softmax(scores, dim=-1), V[g])

        def fill(shape):
            return torch.full(shape, 123.0, device=self.device, dtype=torch.float32)

        return {
            "Q": Q,
            "K": K,
            "V": V,
            "O": out,
            "dO": dO,
            "lse": lse,
            "dQ": fill((num_q_heads, seq_len, head_dim)),
            "dK": fill((num_kv_heads, seq_len, head_dim)),
            "dV": fill((num_kv_heads, seq_len, head_dim)),
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
        }

    def _rand(self, shape, low: float, high: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(low, high)

    def _rand_case(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        seed: int,
        qk_range: float = 2.0,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        Q = self._rand((num_q_heads, seq_len, head_dim), -qk_range, qk_range)
        K = self._rand((num_kv_heads, seq_len, head_dim), -qk_range, qk_range)
        V = self._rand((num_kv_heads, seq_len, head_dim), -1.0, 1.0)
        dO = self._rand((num_q_heads, seq_len, head_dim), -1.0, 1.0)
        return self._case(Q, K, V, dO)

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0], [-1.0, 1.0]]],
            device=self.device,
            dtype=dtype,
        )
        K = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]], device=self.device, dtype=dtype)
        V = torch.tensor([[[1.0, 2.0], [3.0, -1.0], [0.0, 1.0]]], device=self.device, dtype=dtype)
        dO = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], [[0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]],
            device=self.device,
            dtype=dtype,
        )
        return self._case(Q, K, V, dO)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests: List[Dict[str, Any]] = []

        # 1: basic example (same as generate_example_test); two query heads share one KV head
        tests.append(self.generate_example_test())

        # 2: single token, two query heads, one KV head. The only visible key is the token
        # itself, so P = 1, dQ = dK = 0 and dV[0, 0] = dO[0, 0] + dO[1, 0]. Catches a strict
        # (j < i) causal mask, which gives dV = 0, and keeping only one head's dV.
        tests.append(
            self._case(
                torch.tensor(
                    [[[1.0, -2.0, 0.5, 3.0]], [[-1.0, 0.0, 2.0, -0.5]]], device=self.device
                ),
                torch.tensor([[[0.5, 1.0, -1.0, 2.0]]], device=self.device),
                torch.tensor([[[3.0, -2.0, 1.0, 0.5]]], device=self.device),
                torch.tensor(
                    [[[1.0, 2.0, -3.0, 0.5]], [[-0.5, 1.0, 1.0, 2.0]]], device=self.device
                ),
            )
        )

        # 3: Q = K = 0: every visible key has weight 1 / (i + 1), so independently
        # dV[0, j] = sum_h sum_{i >= j} dO[h, i] / (i + 1) and dQ = dK = 0. Catches a missing or
        # off-by-one causal mask and a dV that is not summed over the query heads of the group.
        torch.manual_seed(1)
        tests.append(
            self._case(
                torch.zeros((2, 7, 4), device=self.device, dtype=dtype),
                torch.zeros((1, 7, 4), device=self.device, dtype=dtype),
                self._rand((1, 7, 4), -2.0, 2.0),
                self._rand((2, 7, 4), -1.0, 1.0),
            )
        )

        # 4: all-zero inputs (lse[h, i] = log(i + 1)); every gradient is exactly zero
        tests.append(
            self._case(
                torch.zeros((4, 16, 8), device=self.device, dtype=dtype),
                torch.zeros((2, 16, 8), device=self.device, dtype=dtype),
                torch.zeros((2, 16, 8), device=self.device, dtype=dtype),
                torch.zeros((4, 16, 8), device=self.device, dtype=dtype),
            )
        )

        # 5: multi-head attention (group size 1), negative and mixed values: checks the softmax
        # backward itself (the 1/sqrt(head_dim) scale on dQ and dK, the D = rowsum(dO * O) term)
        tests.append(self._rand_case(4, 4, 32, 16, seed=2))

        # 6: GQA head mapping with distinct KV heads: query head h belongs to KV head
        # h / (num_q_heads / num_kv_heads). Using h % num_kv_heads, or writing dK / dV from only
        # one query head of the group (or racing plain stores between heads), changes dK / dV.
        tests.append(self._rand_case(8, 2, 64, 32, seed=3))

        # 7: multi-query attention (one KV head shared by 8 query heads), non-power-of-two
        # sequence length and head_dim, so tiles over keys, queries and head_dim have tails
        tests.append(self._rand_case(8, 1, 100, 50, seed=4))

        # 8: odd group size (3), sequence length 255 and head_dim 34: tails in every tile
        # dimension and a group size that does not divide a power-of-two thread layout
        tests.append(self._rand_case(6, 2, 255, 34, seed=5))

        # 9: sharp softmax: Q and K entries are +-5, so diagonal scores are 5 * 5 * 16 / 4 = 100
        # and exp(score) overflows float32. Recomputing P as exp(s) / sum exp(s) without
        # subtracting lse (or a running max) yields inf / NaN; exp(s - lse) stays finite.
        torch.manual_seed(6)
        qk = 5.0 * torch.sign(torch.randn((2, 40, 16), device=self.device, dtype=dtype))
        tests.append(
            self._case(
                qk,
                qk[:1].clone(),
                self._rand((1, 40, 16), -1.0, 1.0),
                self._rand((2, 40, 16), -1.0, 1.0),
            )
        )

        # 10: realistic size with the performance-test head layout (Llama 3.2 1B attention)
        torch.manual_seed(7)
        tests.append(
            self._case(
                torch.randn((32, 1024, 64), device=self.device, dtype=dtype),
                torch.randn((8, 1024, 64), device=self.device, dtype=dtype),
                torch.randn((8, 1024, 64), device=self.device, dtype=dtype),
                torch.randn((32, 1024, 64), device=self.device, dtype=dtype),
            )
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        num_q_heads, num_kv_heads, seq_len, head_dim = 32, 8, 2048, 64
        torch.manual_seed(0)
        return self._case(
            torch.randn((num_q_heads, seq_len, head_dim), device=self.device, dtype=dtype),
            torch.randn((num_kv_heads, seq_len, head_dim), device=self.device, dtype=dtype),
            torch.randn((num_kv_heads, seq_len, head_dim), device=self.device, dtype=dtype),
            torch.randn((num_q_heads, seq_len, head_dim), device=self.device, dtype=dtype),
        )
