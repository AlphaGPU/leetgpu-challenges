import ctypes
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase

# Llama-style architecture constants
D = 512  # model dimension
NUM_Q_HEADS = 8  # number of query heads
NUM_KV_HEADS = 2  # number of key/value heads (grouped query attention)
HEAD_DIM = D // NUM_Q_HEADS  # = 64
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # = 512
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # = 128
GQA_GROUPS = NUM_Q_HEADS // NUM_KV_HEADS  # = 4
FFN_HIDDEN = 1408  # SwiGLU intermediate dimension

# Weight buffer layout offsets (all projections stored as (out_dim, in_dim))
O_RMS1_W = 0
O_WQ = O_RMS1_W + D  # Q projection: Q_DIM x D
O_WK = O_WQ + Q_DIM * D  # K projection: KV_DIM x D
O_WV = O_WK + KV_DIM * D  # V projection: KV_DIM x D
O_WO = O_WV + KV_DIM * D  # output projection: D x D
O_RMS2_W = O_WO + D * D  # RMS norm 2 weights: D
O_WGATE = O_RMS2_W + D  # gate projection: FFN_HIDDEN x D
O_WUP = O_WGATE + FFN_HIDDEN * D  # up projection: FFN_HIDDEN x D
O_WDOWN = O_WUP + FFN_HIDDEN * D  # down projection: D x FFN_HIDDEN
TOTAL_WEIGHTS = O_WDOWN + D * FFN_HIDDEN


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Llama Transformer Block",
            atol=1e-03,
            rtol=1e-03,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        weights: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seq_len: int,
    ):
        assert x.shape == (seq_len, D)
        assert output.shape == (seq_len, D)
        assert weights.shape == (TOTAL_WEIGHTS,)
        assert cos.shape == (seq_len, HEAD_DIM // 2)
        assert sin.shape == (seq_len, HEAD_DIM // 2)
        assert x.dtype == output.dtype == weights.dtype == cos.dtype == sin.dtype
        assert x.device.type == "cuda"
        assert output.device.type == "cuda"
        assert weights.device.type == "cuda"
        assert cos.device.type == "cuda"
        assert sin.device.type == "cuda"

        def rms_norm(z, w):
            return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-5) * w

        def apply_rope(qk, c, s):
            # qk: (seq_len, num_heads, head_dim)
            q1, q2 = qk[..., : HEAD_DIM // 2], qk[..., HEAD_DIM // 2 :]
            c = c.unsqueeze(1)  # (seq_len, 1, head_dim//2)
            s = s.unsqueeze(1)
            return torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], dim=-1)

        # unpack weights
        rms1_w = weights[O_RMS1_W:O_WQ]
        W_Q = weights[O_WQ:O_WK].view(Q_DIM, D)
        W_K = weights[O_WK:O_WV].view(KV_DIM, D)
        W_V = weights[O_WV:O_WO].view(KV_DIM, D)
        W_O = weights[O_WO:O_RMS2_W].view(D, D)
        rms2_w = weights[O_RMS2_W:O_WGATE]
        W_gate = weights[O_WGATE:O_WUP].view(FFN_HIDDEN, D)
        W_up = weights[O_WUP:O_WDOWN].view(FFN_HIDDEN, D)
        W_down = weights[O_WDOWN:TOTAL_WEIGHTS].view(D, FFN_HIDDEN)

        # --- Attention sub-block ---
        x_norm = rms_norm(x, rms1_w)

        # QKV projections
        q = (x_norm @ W_Q.T).view(seq_len, NUM_Q_HEADS, HEAD_DIM)
        k = (x_norm @ W_K.T).view(seq_len, NUM_KV_HEADS, HEAD_DIM)
        v = (x_norm @ W_V.T).view(seq_len, NUM_KV_HEADS, HEAD_DIM)

        # Apply RoPE to Q and K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Reshape for batched matmul: (num_heads, seq_len, head_dim)
        q = q.transpose(0, 1)  # (NUM_Q_HEADS, seq_len, HEAD_DIM)
        k = k.transpose(0, 1)  # (NUM_KV_HEADS, seq_len, HEAD_DIM)
        v = v.transpose(0, 1)  # (NUM_KV_HEADS, seq_len, HEAD_DIM)

        # GQA: broadcast K and V to match Q heads
        k = k.repeat_interleave(GQA_GROUPS, dim=0)  # (NUM_Q_HEADS, seq_len, HEAD_DIM)
        v = v.repeat_interleave(GQA_GROUPS, dim=0)

        # Causal scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        scores = scores + causal_mask
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (NUM_Q_HEADS, seq_len, HEAD_DIM)

        # Merge heads and project
        attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, D)
        attn_proj = attn_out @ W_O.T

        # Residual 1
        hidden = x + attn_proj

        # --- FFN sub-block ---
        h_norm = rms_norm(hidden, rms2_w)

        # SwiGLU: gate * up, then project down
        gate = F.silu(h_norm @ W_gate.T)
        up = h_norm @ W_up.T
        ffn_out = (gate * up) @ W_down.T

        # Residual 2
        output.copy_(hidden + ffn_out)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "weights": (ctypes.POINTER(ctypes.c_float), "in"),
            "cos": (ctypes.POINTER(ctypes.c_float), "in"),
            "sin": (ctypes.POINTER(ctypes.c_float), "in"),
            "seq_len": (ctypes.c_int, "in"),
        }

    def _make_rope_tables(self, seq_len, device, dtype, theta=10000.0):
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = 1.0 / (
            theta ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=dtype) / HEAD_DIM)
        )
        angles = torch.outer(positions, freqs)  # (seq_len, HEAD_DIM//2)
        return angles.cos(), angles.sin()

    def _make_weights(self, device, dtype):
        scale = 0.02
        rms1_w = torch.empty(D, device=device, dtype=dtype).uniform_(0.8, 1.2)
        W_Q = torch.empty(Q_DIM, D, device=device, dtype=dtype).normal_(0, scale)
        W_K = torch.empty(KV_DIM, D, device=device, dtype=dtype).normal_(0, scale)
        W_V = torch.empty(KV_DIM, D, device=device, dtype=dtype).normal_(0, scale)
        W_O = torch.empty(D, D, device=device, dtype=dtype).normal_(0, scale)
        rms2_w = torch.empty(D, device=device, dtype=dtype).uniform_(0.8, 1.2)
        W_gate = torch.empty(FFN_HIDDEN, D, device=device, dtype=dtype).normal_(0, scale)
        W_up = torch.empty(FFN_HIDDEN, D, device=device, dtype=dtype).normal_(0, scale)
        W_down = torch.empty(D, FFN_HIDDEN, device=device, dtype=dtype).normal_(0, scale)
        return torch.cat(
            [
                rms1_w,
                W_Q.flatten(),
                W_K.flatten(),
                W_V.flatten(),
                W_O.flatten(),
                rms2_w,
                W_gate.flatten(),
                W_up.flatten(),
                W_down.flatten(),
            ]
        )

    def _make_test_case(self, seq_len, zero_x=False):
        dtype = torch.float32
        device = "cuda"
        weights = self._make_weights(device, dtype)
        cos, sin = self._make_rope_tables(seq_len, device, dtype)
        if zero_x:
            x = torch.zeros(seq_len, D, device=device, dtype=dtype)
        else:
            x = torch.empty(seq_len, D, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        return {
            "x": x,
            "output": torch.empty(seq_len, D, device=device, dtype=dtype),
            "weights": weights,
            "cos": cos,
            "sin": sin,
            "seq_len": seq_len,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test_case(4)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        # single token (decode phase)
        tests.append(self._make_test_case(1))
        # zero input
        tests.append(self._make_test_case(4, zero_x=True))
        # small edge cases
        tests.append(self._make_test_case(2))
        tests.append(self._make_test_case(4))
        # power-of-2
        tests.append(self._make_test_case(16))
        tests.append(self._make_test_case(64))
        # non-power-of-2
        tests.append(self._make_test_case(30))
        tests.append(self._make_test_case(100))
        # realistic inference lengths
        tests.append(self._make_test_case(128))
        tests.append(self._make_test_case(256))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(2048)
