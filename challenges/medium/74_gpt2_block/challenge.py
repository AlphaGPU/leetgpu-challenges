import ctypes
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase

# GPT-2 124M fixed dimensions
D = 768
H = 12
DH = D // H  # 64
FFN = 3072

# Weight layout offsets in the packed buffer
O_LN1_W = 0
O_LN1_B = O_LN1_W + D
O_WQKV = O_LN1_B + D
O_BQKV = O_WQKV + D * 3 * D
O_WAPROJ = O_BQKV + 3 * D
O_BAPROJ = O_WAPROJ + D * D
O_LN2_W = O_BAPROJ + D
O_LN2_B = O_LN2_W + D
O_WFC = O_LN2_B + D
O_BFC = O_WFC + D * FFN
O_WPROJ = O_BFC + FFN
O_BPROJ = O_WPROJ + FFN * D
TOTAL_WEIGHTS = O_BPROJ + D


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
        weights: torch.Tensor,
        seq_len: int,
    ):
        assert x.shape == (seq_len, D)
        assert output.shape == (seq_len, D)
        assert weights.shape == (TOTAL_WEIGHTS,)
        assert x.dtype == output.dtype == weights.dtype
        assert x.device.type == "cuda"
        assert output.device.type == "cuda"
        assert weights.device.type == "cuda"

        # unpack weights
        ln1_w = weights[O_LN1_W:O_LN1_B]
        ln1_b = weights[O_LN1_B:O_WQKV]
        W_qkv = weights[O_WQKV:O_BQKV].view(D, 3 * D)
        b_qkv = weights[O_BQKV:O_WAPROJ]
        W_attn = weights[O_WAPROJ:O_BAPROJ].view(D, D)
        b_attn = weights[O_BAPROJ:O_LN2_W]
        ln2_w = weights[O_LN2_W:O_LN2_B]
        ln2_b = weights[O_LN2_B:O_WFC]
        W_fc = weights[O_WFC:O_BFC].view(D, FFN)
        b_fc = weights[O_BFC:O_WPROJ]
        W_proj = weights[O_WPROJ:O_BPROJ].view(FFN, D)
        b_proj = weights[O_BPROJ : O_BPROJ + D]

        # layer norm 1
        x_norm = F.layer_norm(x, [D], ln1_w, ln1_b, eps=1e-5)

        # qkv projection
        qkv = x_norm @ W_qkv + b_qkv
        q, k, v = qkv.split(D, dim=-1)

        # reshape for multi-head attention: (H, seq_len, DH)
        q = q.view(seq_len, H, DH).transpose(0, 1)
        k = k.view(seq_len, H, DH).transpose(0, 1)
        v = v.view(seq_len, H, DH).transpose(0, 1)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(DH)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # concat heads and project
        attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, D)
        attn_proj = attn_out @ W_attn + b_attn

        # residual connection 1
        hidden = x + attn_proj

        # layer norm 2
        h_norm = F.layer_norm(hidden, [D], ln2_w, ln2_b, eps=1e-5)

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
            "weights": (ctypes.POINTER(ctypes.c_float), "in"),
            "seq_len": (ctypes.c_int, "in"),
        }

    def _make_weights(self, device, dtype):
        scale = 0.02
        ln1_w = torch.empty(D, device=device, dtype=dtype).uniform_(0.8, 1.2)
        ln1_b = torch.empty(D, device=device, dtype=dtype).uniform_(-0.1, 0.1)
        W_qkv = torch.empty(D, 3 * D, device=device, dtype=dtype).normal_(0, scale)
        b_qkv = torch.zeros(3 * D, device=device, dtype=dtype)
        W_attn = torch.empty(D, D, device=device, dtype=dtype).normal_(0, scale)
        b_attn = torch.zeros(D, device=device, dtype=dtype)
        ln2_w = torch.empty(D, device=device, dtype=dtype).uniform_(0.8, 1.2)
        ln2_b = torch.empty(D, device=device, dtype=dtype).uniform_(-0.1, 0.1)
        W_fc = torch.empty(D, FFN, device=device, dtype=dtype).normal_(0, scale)
        b_fc = torch.zeros(FFN, device=device, dtype=dtype)
        W_proj = torch.empty(FFN, D, device=device, dtype=dtype).normal_(0, scale)
        b_proj = torch.zeros(D, device=device, dtype=dtype)
        return torch.cat(
            [
                ln1_w,
                ln1_b,
                W_qkv.flatten(),
                b_qkv,
                W_attn.flatten(),
                b_attn,
                ln2_w,
                ln2_b,
                W_fc.flatten(),
                b_fc,
                W_proj.flatten(),
                b_proj,
            ]
        )

    def _make_test_case(self, seq_len, zero_x=False):
        dtype = torch.float32
        device = "cuda"
        weights = self._make_weights(device, dtype)
        if zero_x:
            x = torch.zeros(seq_len, D, device=device, dtype=dtype)
        else:
            x = torch.empty(seq_len, D, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        return {
            "x": x,
            "output": torch.empty(seq_len, D, device=device, dtype=dtype),
            "weights": weights,
            "seq_len": seq_len,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test_case(4)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        # single token
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
        # realistic
        tests.append(self._make_test_case(128))
        tests.append(self._make_test_case(256))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(1024)
