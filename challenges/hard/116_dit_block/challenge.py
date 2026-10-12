import ctypes
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase

# DiT block dimensions
D = 512  # hidden size
H = 8  # number of attention heads
DH = D // H  # 64
MLP = 4 * D  # 2048
EPS = 1e-6

# Weight layout offsets in the packed buffer (matrices stored row-major as (out_dim, in_dim))
O_WADA = 0
O_BADA = O_WADA + 6 * D * D
O_WQKV = O_BADA + 6 * D
O_BQKV = O_WQKV + 3 * D * D
O_WO = O_BQKV + 3 * D
O_BO = O_WO + D * D
O_WFC1 = O_BO + D
O_BFC1 = O_WFC1 + MLP * D
O_WFC2 = O_BFC1 + MLP
O_BFC2 = O_WFC2 + D * MLP
TOTAL_WEIGHTS = O_BFC2 + D


class Challenge(ChallengeBase):
    name = "Diffusion Transformer Block"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        output: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ):
        assert x.shape == (batch_size, seq_len, D)
        assert output.shape == (batch_size, seq_len, D)
        assert c.shape == (batch_size, D)
        assert weights.shape == (TOTAL_WEIGHTS,)
        assert x.dtype == c.dtype == output.dtype == weights.dtype

        # unpack weights
        W_ada = weights[O_WADA:O_BADA].view(6 * D, D)
        b_ada = weights[O_BADA:O_WQKV]
        W_qkv = weights[O_WQKV:O_BQKV].view(3 * D, D)
        b_qkv = weights[O_BQKV:O_WO]
        W_o = weights[O_WO:O_BO].view(D, D)
        b_o = weights[O_BO:O_WFC1]
        W_fc1 = weights[O_WFC1:O_BFC1].view(MLP, D)
        b_fc1 = weights[O_BFC1:O_WFC2]
        W_fc2 = weights[O_WFC2:O_BFC2].view(D, MLP)
        b_fc2 = weights[O_BFC2:TOTAL_WEIGHTS]

        # adaLN modulation: six (batch_size, D) parameter vectors per sample
        mod = F.silu(c) @ W_ada.T + b_ada
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.view(
            batch_size, 6, D
        ).unbind(dim=1)

        # --- attention sub-block ---
        h = F.layer_norm(x, [D], eps=EPS)
        h = h * (1.0 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        qkv = h @ W_qkv.T + b_qkv
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(batch_size, seq_len, H, DH).transpose(1, 2)
        k = k.view(batch_size, seq_len, H, DH).transpose(1, 2)
        v = v.view(batch_size, seq_len, H, DH).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(DH)
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, D)
        attn = attn @ W_o.T + b_o

        x1 = x + gate_msa.unsqueeze(1) * attn

        # --- mlp sub-block ---
        h2 = F.layer_norm(x1, [D], eps=EPS)
        h2 = h2 * (1.0 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        mlp = F.gelu(h2 @ W_fc1.T + b_fc1, approximate="tanh") @ W_fc2.T + b_fc2

        output.copy_(x1 + gate_mlp.unsqueeze(1) * mlp)

    def reference_impl_jax(self, x, c, weights, batch_size, seq_len):
        import jax
        import jax.numpy as jnp

        def layer_norm(z):
            mean = jnp.mean(z, axis=-1, keepdims=True)
            var = jnp.mean((z - mean) ** 2, axis=-1, keepdims=True)
            return (z - mean) * jax.lax.rsqrt(var + EPS)

        # unpack weights
        W_ada = weights[O_WADA:O_BADA].reshape(6 * D, D)
        b_ada = weights[O_BADA:O_WQKV]
        W_qkv = weights[O_WQKV:O_BQKV].reshape(3 * D, D)
        b_qkv = weights[O_BQKV:O_WO]
        W_o = weights[O_WO:O_BO].reshape(D, D)
        b_o = weights[O_BO:O_WFC1]
        W_fc1 = weights[O_WFC1:O_BFC1].reshape(MLP, D)
        b_fc1 = weights[O_BFC1:O_WFC2]
        W_fc2 = weights[O_WFC2:O_BFC2].reshape(D, MLP)
        b_fc2 = weights[O_BFC2:TOTAL_WEIGHTS]

        # adaLN modulation: six (batch_size, D) parameter vectors per sample
        mod = jax.nn.silu(c) @ W_ada.T + b_ada
        mod = mod.reshape(batch_size, 6, D)
        shift_msa = mod[:, 0]
        scale_msa = mod[:, 1]
        gate_msa = mod[:, 2]
        shift_mlp = mod[:, 3]
        scale_mlp = mod[:, 4]
        gate_mlp = mod[:, 5]

        # --- attention sub-block ---
        h = layer_norm(x)
        h = h * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        qkv = h @ W_qkv.T + b_qkv
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(batch_size, seq_len, H, DH).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, H, DH).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, H, DH).transpose(0, 2, 1, 3)

        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(DH)
        attn = jnp.matmul(jax.nn.softmax(scores, axis=-1), v)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, D)
        attn = attn @ W_o.T + b_o

        x1 = x + gate_msa[:, None, :] * attn

        # --- mlp sub-block ---
        h2 = layer_norm(x1)
        h2 = h2 * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]

        mlp = jax.nn.gelu(h2 @ W_fc1.T + b_fc1, approximate=True) @ W_fc2.T + b_fc2

        return x1 + gate_mlp[:, None, :] * mlp

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "c": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "weights": (ctypes.POINTER(ctypes.c_float), "in"),
            "batch_size": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
        }

    def _make_weights(self, dtype):
        device = self.device
        scale = 0.02
        return torch.cat(
            [
                torch.empty(6 * D, D, device=device, dtype=dtype).normal_(0, scale).flatten(),
                torch.empty(6 * D, device=device, dtype=dtype).uniform_(-0.1, 0.1),
                torch.empty(3 * D, D, device=device, dtype=dtype).normal_(0, scale).flatten(),
                torch.zeros(3 * D, device=device, dtype=dtype),
                torch.empty(D, D, device=device, dtype=dtype).normal_(0, scale).flatten(),
                torch.zeros(D, device=device, dtype=dtype),
                torch.empty(MLP, D, device=device, dtype=dtype).normal_(0, scale).flatten(),
                torch.zeros(MLP, device=device, dtype=dtype),
                torch.empty(D, MLP, device=device, dtype=dtype).normal_(0, scale).flatten(),
                torch.zeros(D, device=device, dtype=dtype),
            ]
        )

    def _make_test_case(self, batch_size, seq_len, zero_x=False, zero_c=False):
        dtype = torch.float32
        device = self.device
        if zero_x:
            x = torch.zeros(batch_size, seq_len, D, device=device, dtype=dtype)
        else:
            x = torch.empty(batch_size, seq_len, D, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        if zero_c:
            c = torch.zeros(batch_size, D, device=device, dtype=dtype)
        else:
            c = torch.empty(batch_size, D, device=device, dtype=dtype).uniform_(-2.0, 2.0)
        return {
            "x": x,
            "c": c,
            "output": torch.empty(batch_size, seq_len, D, device=device, dtype=dtype),
            "weights": self._make_weights(dtype),
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        return self._make_test_case(1, 4)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(1)
        tests = []
        # edge: single token, single sample
        tests.append(self._make_test_case(1, 1))
        # edge: zero hidden states
        tests.append(self._make_test_case(2, 3, zero_x=True))
        # edge: zero conditioning vector
        tests.append(self._make_test_case(1, 4, zero_c=True))
        # power-of-2 sequence lengths
        tests.append(self._make_test_case(2, 16))
        tests.append(self._make_test_case(4, 64))
        # non-power-of-2
        tests.append(self._make_test_case(3, 30))
        tests.append(self._make_test_case(2, 100))
        tests.append(self._make_test_case(5, 255))
        # realistic latent grids (16x16 and 14x14 patch grids)
        tests.append(self._make_test_case(2, 256))
        tests.append(self._make_test_case(8, 196))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(2)
        return self._make_test_case(4, 1024)
