import ctypes
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase

# Model architecture constants
VOCAB_SIZE = 10
MODEL_DIM = 2
HEAD_DIM = 2
PROMPT_LEN = 31
OUTPUT_DIGITS = 11
RMS_EPS = 1e-6

# Derived constants from the hand-crafted 10-parameter adder model
EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(MODEL_DIM)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
DECODE_QUAD = 1e-3
DECODE_CURVATURE = 0.1
ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)
TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
CARRY_ALPHA = 256.0 / CONST_NORM
ATTN_SCALE = (HEAD_DIM**-0.5) * (QK_NORM_SCALE**2)

# Weight buffer layout (10 parameters total)
O_EMBED = 0  # [2] embedding: e(d) = [w0 - w1*d^2, -d]
O_QPROJ = 2  # [2] Q projection weights
O_VPROJ = 4  # [1] V projection weight
O_GATE = 5  # [2] MLP gate weights
O_CARRY = 7  # [1] MLP carry weight
O_NORM = 8  # [2] final RMSNorm weight
TOTAL_WEIGHTS = 10


def _encode_pair(a: int, b: int) -> list:
    a_digits = [int(c) for c in f"{a:010d}"][::-1]
    b_digits = [int(c) for c in f"{b:010d}"][::-1]
    return [0] + a_digits + [0] * 9 + b_digits + [0]


def _encode_pairs_batch(a_vals: torch.Tensor, b_vals: torch.Tensor, device) -> torch.Tensor:
    batch_size = a_vals.shape[0]
    prompts = torch.zeros(batch_size, PROMPT_LEN, device=device, dtype=torch.int32)
    a = a_vals.clone().to(torch.int64)
    for i in range(10):
        prompts[:, 1 + i] = (a % 10).to(torch.int32)
        a = a // 10
    b = b_vals.clone().to(torch.int64)
    for i in range(10):
        prompts[:, 20 + i] = (b % 10).to(torch.int32)
        b = b // 10
    return prompts


def _init_weights(device) -> torch.Tensor:
    w = torch.zeros(TOTAL_WEIGHTS, device=device, dtype=torch.float32)
    w[O_EMBED] = EMBED_CONST
    w[O_EMBED + 1] = DECODE_QUAD
    w[O_QPROJ] = math.cos(PHI)
    w[O_QPROJ + 1] = -math.sin(PHI)
    w[O_VPROJ] = -22.0 * DIGIT_SCALE
    w[O_GATE] = CARRY_ALPHA * (-94.0) / CONST_NORM
    w[O_GATE + 1] = CARRY_ALPHA * DIGIT_SCALE
    w[O_CARRY] = (100.0 / CARRY_ALPHA) * (1.0 / CONST_NORM)
    w[O_NORM] = (DECODE_CURVATURE / DECODE_QUAD) / CONST_NORM
    w[O_NORM + 1] = -(DIGIT_SCALE / 50.0)
    return w


def _unit_rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + RMS_EPS)


def _forward_pass(seq: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = seq.shape
    device = seq.device

    embed_w = weights[O_EMBED : O_EMBED + 2]
    q_w = weights[O_QPROJ : O_QPROJ + 2]
    v_w = weights[O_VPROJ]
    gate_w = weights[O_GATE : O_GATE + 2]
    carry_w = weights[O_CARRY]
    norm_w = weights[O_NORM : O_NORM + 2]

    digits = torch.arange(VOCAB_SIZE, device=device, dtype=torch.float32)
    embed_table = torch.stack(
        [embed_w[0] - embed_w[1] * digits * digits, -digits], dim=-1
    )  # [10, 2]

    h = embed_table[seq.long()]  # [batch, seq_len, 2]

    # Pre-attention unit RMSNorm (no learned parameters)
    h_norm = _unit_rms_norm(h)

    # Q projection: [h0*qw0, h0*qw1]
    q = torch.stack([h_norm[..., 0] * q_w[0], h_norm[..., 0] * q_w[1]], dim=-1)

    # K projection: [h0, 0]
    k = torch.stack([h_norm[..., 0], torch.zeros_like(h_norm[..., 0])], dim=-1)

    # V projection: [h1*vw, 0]
    v = torch.stack([h_norm[..., 1] * v_w, torch.zeros_like(h_norm[..., 1])], dim=-1)

    # QK norm
    q = _unit_rms_norm(q)
    k = _unit_rms_norm(k)

    # RoPE
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = positions * OMEGA
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    q_rot = torch.stack(
        [q[..., 0] * cos_a - q[..., 1] * sin_a, q[..., 0] * sin_a + q[..., 1] * cos_a], dim=-1
    )
    k_rot = torch.stack(
        [k[..., 0] * cos_a - k[..., 1] * sin_a, k[..., 0] * sin_a + k[..., 1] * cos_a], dim=-1
    )

    # Attention: [batch, 1, seq_len, 2]
    q_rot = q_rot.unsqueeze(1)
    k_rot = k_rot.unsqueeze(1)
    v = v.unsqueeze(1)

    attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * ATTN_SCALE
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_probs, v).squeeze(1)  # [batch, seq_len, 2]

    # O projection: [0, attn[..., 0]]
    o = torch.stack([torch.zeros_like(attn_out[..., 0]), attn_out[..., 0]], dim=-1)

    # Residual
    h = h + o

    # Pre-MLP unit RMSNorm
    h_norm2 = _unit_rms_norm(h)

    # MLP gate projection
    a_gate = gate_w[0]
    c_gate = gate_w[1]
    g0 = h_norm2[..., 0] * a_gate + h_norm2[..., 1] * c_gate
    g1 = h_norm2[..., 0] * (a_gate - c_gate / EMBED_CONST) + h_norm2[..., 1] * c_gate
    gate = torch.stack([g0, g1], dim=-1)

    # MLP carry projection with SwiGLU
    base = h_norm2[..., 0]
    up = base.unsqueeze(-1).expand_as(gate)
    mix = F.silu(gate) * up
    mlp_out = torch.stack([torch.zeros_like(base), carry_w * (mix[..., 1] - mix[..., 0])], dim=-1)

    # Residual
    h = h + mlp_out

    # Final RMSNorm (with learned weight)
    rms = torch.sqrt(torch.mean(h * h, dim=-1, keepdim=True) + RMS_EPS)
    h = (h / rms) * norm_w

    # Output projection (tied with embedding)
    logits = h @ embed_table.T  # [batch, seq_len, 10]
    return logits


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Adder Transformer Inference",
            atol=1e-2,
            rtol=1e-2,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        prompts: torch.Tensor,
        output: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
    ):
        assert prompts.shape == (batch_size, PROMPT_LEN)
        assert prompts.dtype == torch.int32
        assert prompts.device.type == "cuda"
        assert output.shape == (batch_size, OUTPUT_DIGITS, VOCAB_SIZE)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"
        assert weights.shape == (TOTAL_WEIGHTS,)
        assert weights.dtype == torch.float32
        assert weights.device.type == "cuda"

        seq = prompts.clone()
        for step in range(OUTPUT_DIGITS):
            logits = _forward_pass(seq, weights)
            last_logits = logits[:, -1, :]
            output[:, step, :] = last_logits
            next_token = last_logits.argmax(dim=-1).to(torch.int32)
            seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "prompts": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "weights": (ctypes.POINTER(ctypes.c_float), "in"),
            "batch_size": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        pairs = [(3, 5), (99, 1)]
        batch_size = len(pairs)
        prompts = torch.tensor(
            [_encode_pair(a, b) for a, b in pairs],
            device=device,
            dtype=torch.int32,
        )
        weights = _init_weights(device)
        output = torch.zeros(
            batch_size, OUTPUT_DIGITS, VOCAB_SIZE, device=device, dtype=torch.float32
        )
        return {
            "prompts": prompts,
            "output": output,
            "weights": weights,
            "batch_size": batch_size,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        device = "cuda"
        tests = []

        def _make_test(pairs):
            batch_size = len(pairs)
            prompts = torch.tensor(
                [_encode_pair(a, b) for a, b in pairs],
                device=device,
                dtype=torch.int32,
            )
            weights = _init_weights(device)
            output = torch.zeros(
                batch_size, OUTPUT_DIGITS, VOCAB_SIZE, device=device, dtype=torch.float32
            )
            return {
                "prompts": prompts,
                "output": output,
                "weights": weights,
                "batch_size": batch_size,
            }

        # Edge: single pair, both zero
        tests.append(_make_test([(0, 0)]))

        # Edge: single pair, max carry propagation
        tests.append(_make_test([(9999999999, 1)]))

        # Edge: small batch, simple sums
        tests.append(_make_test([(1, 2), (3, 4)]))

        # Power-of-2 batch: 16
        torch.manual_seed(42)
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(16)
                ]
            )
        )

        # Power-of-2 batch: 64
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(64)
                ]
            )
        )

        # Non-power-of-2: 30
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(30)
                ]
            )
        )

        # Non-power-of-2: 100
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(100)
                ]
            )
        )

        # Realistic: 1000
        tests.append(
            _make_test(
                [
                    (torch.randint(0, 10**10, (1,)).item(), torch.randint(0, 10**10, (1,)).item())
                    for _ in range(1000)
                ]
            )
        )

        # All zeros
        tests.append(_make_test([(0, 0)] * 8))

        # Max values
        tests.append(_make_test([(9999999999, 9999999999)] * 4))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        device = "cuda"
        batch_size = 100000
        torch.manual_seed(123)
        a_vals = torch.randint(0, 10**10, (batch_size,), dtype=torch.int64)
        b_vals = torch.randint(0, 10**10, (batch_size,), dtype=torch.int64)
        prompts = _encode_pairs_batch(a_vals, b_vals, device)
        weights = _init_weights(device)
        output = torch.zeros(
            batch_size, OUTPUT_DIGITS, VOCAB_SIZE, device=device, dtype=torch.float32
        )
        return {
            "prompts": prompts,
            "output": output,
            "weights": weights,
            "batch_size": batch_size,
        }
