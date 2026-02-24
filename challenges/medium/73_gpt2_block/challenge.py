import ctypes
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="GPT-2 Transformer Block",
            atol=1e-03,
            rtol=1e-03,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        ln1_weight: torch.Tensor,
        ln1_bias: torch.Tensor,
        W_qkv: torch.Tensor,
        b_qkv: torch.Tensor,
        W_attn_proj: torch.Tensor,
        b_attn_proj: torch.Tensor,
        ln2_weight: torch.Tensor,
        ln2_bias: torch.Tensor,
        W_fc: torch.Tensor,
        b_fc: torch.Tensor,
        W_proj: torch.Tensor,
        b_proj: torch.Tensor,
        seq_len: int,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
    ):
        assert x.shape == (seq_len, d_model)
        assert output.shape == (seq_len, d_model)
        assert ln1_weight.shape == (d_model,)
        assert ln1_bias.shape == (d_model,)
        assert W_qkv.shape == (d_model, 3 * d_model)
        assert b_qkv.shape == (3 * d_model,)
        assert W_attn_proj.shape == (d_model, d_model)
        assert b_attn_proj.shape == (d_model,)
        assert ln2_weight.shape == (d_model,)
        assert ln2_bias.shape == (d_model,)
        assert W_fc.shape == (d_model, ffn_dim)
        assert b_fc.shape == (ffn_dim,)
        assert W_proj.shape == (ffn_dim, d_model)
        assert b_proj.shape == (d_model,)
        assert x.dtype == output.dtype
        assert x.device == output.device
        assert d_model % n_heads == 0

        d_head = d_model // n_heads

        # layer norm 1
        x_norm = F.layer_norm(x, [d_model], ln1_weight, ln1_bias, eps=1e-5)

        # qkv projection
        qkv = x_norm @ W_qkv + b_qkv
        q, k, v = qkv.split(d_model, dim=-1)

        # reshape for multi-head attention: (n_heads, seq_len, d_head)
        q = q.view(seq_len, n_heads, d_head).transpose(0, 1)
        k = k.view(seq_len, n_heads, d_head).transpose(0, 1)
        v = v.view(seq_len, n_heads, d_head).transpose(0, 1)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # concat heads and project
        attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, d_model)
        attn_proj = attn_out @ W_attn_proj + b_attn_proj

        # residual connection 1
        hidden = x + attn_proj

        # layer norm 2
        h_norm = F.layer_norm(hidden, [d_model], ln2_weight, ln2_bias, eps=1e-5)

        # ffn: linear -> gelu (tanh approx) -> linear
        fc = h_norm @ W_fc + b_fc
        fc = F.gelu(fc, approximate="tanh")
        proj = fc @ W_proj + b_proj

        # residual connection 2
        output.copy_(hidden + proj)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "ln1_weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "ln1_bias": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_qkv": (ctypes.POINTER(ctypes.c_float), "in"),
            "b_qkv": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_attn_proj": (ctypes.POINTER(ctypes.c_float), "in"),
            "b_attn_proj": (ctypes.POINTER(ctypes.c_float), "in"),
            "ln2_weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "ln2_bias": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_fc": (ctypes.POINTER(ctypes.c_float), "in"),
            "b_fc": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_proj": (ctypes.POINTER(ctypes.c_float), "in"),
            "b_proj": (ctypes.POINTER(ctypes.c_float), "in"),
            "seq_len": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "n_heads": (ctypes.c_int, "in"),
            "ffn_dim": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, seq_len):
        dtype = torch.float32
        device = "cuda"
        d_model = 768
        n_heads = 12
        ffn_dim = 3072
        scale = 0.02
        return {
            "x": torch.empty(
                seq_len, d_model, device=device, dtype=dtype
            ).uniform_(-1.0, 1.0),
            "output": torch.empty(seq_len, d_model, device=device, dtype=dtype),
            "ln1_weight": torch.empty(d_model, device=device, dtype=dtype).uniform_(
                0.8, 1.2
            ),
            "ln1_bias": torch.empty(d_model, device=device, dtype=dtype).uniform_(
                -0.1, 0.1
            ),
            "W_qkv": torch.empty(
                d_model, 3 * d_model, device=device, dtype=dtype
            ).normal_(0, scale),
            "b_qkv": torch.zeros(3 * d_model, device=device, dtype=dtype),
            "W_attn_proj": torch.empty(
                d_model, d_model, device=device, dtype=dtype
            ).normal_(0, scale),
            "b_attn_proj": torch.zeros(d_model, device=device, dtype=dtype),
            "ln2_weight": torch.empty(d_model, device=device, dtype=dtype).uniform_(
                0.8, 1.2
            ),
            "ln2_bias": torch.empty(d_model, device=device, dtype=dtype).uniform_(
                -0.1, 0.1
            ),
            "W_fc": torch.empty(
                d_model, ffn_dim, device=device, dtype=dtype
            ).normal_(0, scale),
            "b_fc": torch.zeros(ffn_dim, device=device, dtype=dtype),
            "W_proj": torch.empty(
                ffn_dim, d_model, device=device, dtype=dtype
            ).normal_(0, scale),
            "b_proj": torch.zeros(d_model, device=device, dtype=dtype),
            "seq_len": seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "ffn_dim": ffn_dim,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._make_test_case(4)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        # single token
        tests.append(self._make_test_case(1))
        # small edge cases
        tests.append(self._make_test_case(2))
        tests.append(self._make_test_case(3))
        tests.append(self._make_test_case(4))
        # power-of-2
        tests.append(self._make_test_case(16))
        tests.append(self._make_test_case(64))
        # non-power-of-2
        tests.append(self._make_test_case(30))
        tests.append(self._make_test_case(100))
        # realistic
        tests.append(self._make_test_case(128))
        tests.append(self._make_test_case(256))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(1024)
