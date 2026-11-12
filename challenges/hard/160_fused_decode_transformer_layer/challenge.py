import ctypes
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase

# RMSNorm epsilon, matching Llama-style blocks in this repo.
EPS = 1e-5


def packed_weight_size(d_model: int, H_kv: int, D: int, d_ffn: int) -> int:
    # d_model == H_q * D, so W_q and W_o are both d_model x d_model.
    return (
        d_model
        + d_model * d_model  # W_q
        + d_model * H_kv * D  # W_k
        + d_model * H_kv * D  # W_v
        + d_model * d_model  # W_o
        + d_model  # rms2_w
        + d_model * d_ffn  # W_gate
        + d_model * d_ffn  # W_up
        + d_ffn * d_model  # W_down
    )


def unpack_weights(weights: torch.Tensor, d_model: int, H_q: int, H_kv: int, D: int, d_ffn: int):
    offset = 0

    def take(n: int, shape=None):
        nonlocal offset
        sl = weights[offset : offset + n]
        offset += n
        return sl if shape is None else sl.view(*shape)

    rms1_w = take(d_model)
    W_q = take(d_model * H_q * D, (d_model, H_q * D))
    W_k = take(d_model * H_kv * D, (d_model, H_kv * D))
    W_v = take(d_model * H_kv * D, (d_model, H_kv * D))
    W_o = take(d_model * d_model, (d_model, d_model))
    rms2_w = take(d_model)
    W_gate = take(d_model * d_ffn, (d_model, d_ffn))
    W_up = take(d_model * d_ffn, (d_model, d_ffn))
    W_down = take(d_ffn * d_model, (d_ffn, d_model))
    assert offset == weights.numel()
    return rms1_w, W_q, W_k, W_v, W_o, rms2_w, W_gate, W_up, W_down


class Challenge(ChallengeBase):
    name = "Fused Decode Transformer Layer"
    # One decode step is RMSNorm, several GEMMs, RoPE, softmax attention over the cache, and
    # SwiGLU. With float32 and the ranges used here, a float64 recomputation of the example and
    # of a mid-size functional case differs by well under 1e-5. atol = rtol = 1e-4 leaves room
    # for other float32 reduction orders without accepting a skipped residual, a wrong cache
    # length, dropped GQA repeats, or RoPE applied to V.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        output: torch.Tensor,
        B: int,
        d_model: int,
        H_q: int,
        H_kv: int,
        D: int,
        d_ffn: int,
        S_max: int,
    ):
        assert x.shape == (B, d_model)
        assert weights.shape == (packed_weight_size(d_model, H_kv, D, d_ffn),)
        assert cos_sin_cache.shape == (S_max, D)
        assert positions.shape == (B,)
        assert K_cache.shape == (B, H_kv, S_max, D)
        assert V_cache.shape == (B, H_kv, S_max, D)
        assert output.shape == (B, d_model)
        assert d_model == H_q * D
        assert H_q % H_kv == 0
        assert (
            x.dtype
            == weights.dtype
            == cos_sin_cache.dtype
            == K_cache.dtype
            == V_cache.dtype
            == output.dtype
            == torch.float32
        )
        assert positions.dtype == torch.int32

        rms1_w, W_q, W_k, W_v, W_o, rms2_w, W_gate, W_up, W_down = unpack_weights(
            weights, d_model, H_q, H_kv, D, d_ffn
        )

        def rms_norm(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + EPS) * w

        x_norm = rms_norm(x, rms1_w)
        q = (x_norm @ W_q).view(B, H_q, D)
        k = (x_norm @ W_k).view(B, H_kv, D)
        v = (x_norm @ W_v).view(B, H_kv, D)

        pos = positions.long()
        cos_sin = cos_sin_cache[pos]
        cos = cos_sin[:, : D // 2].unsqueeze(1)
        sin = cos_sin[:, D // 2 :].unsqueeze(1)

        def rope(t: torch.Tensor) -> torch.Tensor:
            t1 = t[..., : D // 2]
            t2 = t[..., D // 2 :]
            return torch.cat((t1 * cos - t2 * sin, t2 * cos + t1 * sin), dim=-1)

        q = rope(q)
        k = rope(k)

        b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, H_kv)
        h_idx = torch.arange(H_kv, device=x.device).unsqueeze(0).expand(B, H_kv)
        p_idx = pos.unsqueeze(1).expand(B, H_kv)
        K_cache[b_idx, h_idx, p_idx] = k
        V_cache[b_idx, h_idx, p_idx] = v

        groups = H_q // H_kv
        k_all = K_cache.repeat_interleave(groups, dim=1)
        v_all = V_cache.repeat_interleave(groups, dim=1)

        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q.unsqueeze(2), k_all.transpose(-2, -1)).squeeze(2) * scale
        col = torch.arange(S_max, device=x.device)
        invalid = col.unsqueeze(0) > pos.unsqueeze(1)
        scores = scores.masked_fill(invalid.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn.unsqueeze(2), v_all).squeeze(2).reshape(B, d_model)

        hidden = x + attn_out @ W_o
        h_norm = rms_norm(hidden, rms2_w)
        gate = F.silu(h_norm @ W_gate)
        up = h_norm @ W_up
        output.copy_(hidden + (gate * up) @ W_down)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "weights": (ctypes.POINTER(ctypes.c_float), "in"),
            "cos_sin_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "positions": (ctypes.POINTER(ctypes.c_int), "in"),
            "K_cache": (ctypes.POINTER(ctypes.c_float), "inout"),
            "V_cache": (ctypes.POINTER(ctypes.c_float), "inout"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "H_q": (ctypes.c_int, "in"),
            "H_kv": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "d_ffn": (ctypes.c_int, "in"),
            "S_max": (ctypes.c_int, "in"),
        }

    def _rope_table(self, S_max: int, D: int) -> torch.Tensor:
        theta = 10000.0
        exponent = torch.arange(0, D, 2, device=self.device, dtype=torch.float32) / D
        inv_freq = 1.0 / (theta**exponent)
        t = torch.arange(S_max, device=self.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        return torch.cat((torch.cos(freqs), torch.sin(freqs)), dim=-1)

    def _pack(
        self,
        rms1_w: torch.Tensor,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
        W_o: torch.Tensor,
        rms2_w: torch.Tensor,
        W_gate: torch.Tensor,
        W_up: torch.Tensor,
        W_down: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                rms1_w,
                W_q.reshape(-1),
                W_k.reshape(-1),
                W_v.reshape(-1),
                W_o.reshape(-1),
                rms2_w,
                W_gate.reshape(-1),
                W_up.reshape(-1),
                W_down.reshape(-1),
            ]
        )

    def _make_test_case(
        self,
        B: int,
        d_model: int,
        H_q: int,
        H_kv: int,
        D: int,
        d_ffn: int,
        S_max: int,
        seed: int,
        positions=None,
        zero_x: bool = False,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        device = self.device
        dtype = torch.float32
        if zero_x:
            x = torch.zeros(B, d_model, device=device, dtype=dtype)
        else:
            x = torch.randn(B, d_model, device=device, dtype=dtype) * 0.1
        rms1_w = torch.empty(d_model, device=device, dtype=dtype).uniform_(0.8, 1.2)
        W_q = torch.randn(d_model, H_q * D, device=device, dtype=dtype) * 0.02
        W_k = torch.randn(d_model, H_kv * D, device=device, dtype=dtype) * 0.02
        W_v = torch.randn(d_model, H_kv * D, device=device, dtype=dtype) * 0.02
        W_o = torch.randn(d_model, d_model, device=device, dtype=dtype) * 0.02
        rms2_w = torch.empty(d_model, device=device, dtype=dtype).uniform_(0.8, 1.2)
        W_gate = torch.randn(d_model, d_ffn, device=device, dtype=dtype) * 0.02
        W_up = torch.randn(d_model, d_ffn, device=device, dtype=dtype) * 0.02
        W_down = torch.randn(d_ffn, d_model, device=device, dtype=dtype) * 0.02
        if positions is None:
            positions = torch.full((B,), S_max - 1, device=device, dtype=torch.int32)
        else:
            positions = torch.as_tensor(positions, device=device, dtype=torch.int32)
        # Prefill the cache with distinctive values so overwriting the wrong slots, or
        # attending to padding, is visible in the output and in the returned cache.
        K_cache = torch.randn(B, H_kv, S_max, D, device=device, dtype=dtype) * 0.1
        V_cache = torch.randn(B, H_kv, S_max, D, device=device, dtype=dtype) * 0.1
        output = torch.empty(B, d_model, device=device, dtype=dtype)
        return {
            "x": x,
            "weights": self._pack(rms1_w, W_q, W_k, W_v, W_o, rms2_w, W_gate, W_up, W_down),
            "cos_sin_cache": self._rope_table(S_max, D),
            "positions": positions,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "output": output,
            "B": B,
            "d_model": d_model,
            "H_q": H_q,
            "H_kv": H_kv,
            "D": D,
            "d_ffn": d_ffn,
            "S_max": S_max,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        B, d_model, H_q, H_kv, D, d_ffn, S_max = 1, 4, 2, 1, 2, 4, 2
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
        rms1_w = torch.ones(d_model, device=device, dtype=dtype)
        W_q = torch.eye(d_model, device=device, dtype=dtype)
        W_k = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        W_v = torch.tensor(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        W_o = torch.eye(d_model, device=device, dtype=dtype)
        rms2_w = torch.ones(d_model, device=device, dtype=dtype)
        W_gate = torch.zeros(d_model, d_ffn, device=device, dtype=dtype)
        W_gate[0, 0] = 1.0
        W_up = torch.zeros(d_model, d_ffn, device=device, dtype=dtype)
        W_up[0, 0] = 1.0
        W_down = torch.zeros(d_ffn, d_model, device=device, dtype=dtype)
        W_down[0, 0] = 1.0
        # pos 0: identity rotation; pos 1: 90-degree rotation of the single RoPE pair.
        cos_sin_cache = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)
        positions = torch.tensor([1], device=device, dtype=torch.int32)
        K_cache = torch.tensor([[[[2.0, 0.0], [9.0, 9.0]]]], device=device, dtype=dtype)
        V_cache = torch.tensor([[[[0.0, 1.0], [8.0, 8.0]]]], device=device, dtype=dtype)
        output = torch.empty(B, d_model, device=device, dtype=dtype)
        return {
            "x": x,
            "weights": self._pack(rms1_w, W_q, W_k, W_v, W_o, rms2_w, W_gate, W_up, W_down),
            "cos_sin_cache": cos_sin_cache,
            "positions": positions,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "output": output,
            "B": B,
            "d_model": d_model,
            "H_q": H_q,
            "H_kv": H_kv,
            "D": D,
            "d_ffn": d_ffn,
            "S_max": S_max,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Independently computed example (loop-based, float64 then compared in float32).
        # Catches a wrong RMSNorm (mean-subtraction or eps), RoPE applied to V, attending to
        # the pre-update garbage in slot 1, skipping GQA head repeats, or dropping a residual.
        example = self.generate_example_test()
        expected_output = torch.tensor(
            [[5.899476051330566, 0.05581318959593773, 0.9999799728393555, 0.5]],
            device=self.device,
            dtype=torch.float32,
        )
        expected_k = torch.tensor(
            [[[[2.0, 0.0], [0.0, 1.999959945678711]]]],
            device=self.device,
            dtype=torch.float32,
        )
        expected_v = torch.tensor(
            [[[[0.0, 1.0], [1.999959945678711, 0.0]]]],
            device=self.device,
            dtype=torch.float32,
        )
        k_before = example["K_cache"].clone()
        v_before = example["V_cache"].clone()
        out = torch.empty_like(example["output"])
        self.reference_impl(
            example["x"],
            example["weights"],
            example["cos_sin_cache"],
            example["positions"],
            example["K_cache"],
            example["V_cache"],
            out,
            example["B"],
            example["d_model"],
            example["H_q"],
            example["H_kv"],
            example["D"],
            example["d_ffn"],
            example["S_max"],
        )
        assert torch.allclose(out, expected_output, atol=1e-5, rtol=1e-5), out
        assert torch.allclose(example["K_cache"], expected_k, atol=1e-6, rtol=1e-6)
        assert torch.allclose(example["V_cache"], expected_v, atol=1e-6, rtol=1e-6)
        # Restore the example's entry cache; the runner starts from the returned buffers.
        example["K_cache"].copy_(k_before)
        example["V_cache"].copy_(v_before)
        tests.append(example)

        # First-token decode: position 0, attend only to the new key. Catches a hardcoded
        # full-cache length or leftover padding terms.
        tests.append(self._make_test_case(1, 4, 2, 1, 2, 4, 4, seed=0, positions=[0]))

        # Two sequences with different cache lengths (0 and 3 of S_max=4). Catches using one
        # shared length for the whole batch or attending past the shorter sequence.
        tests.append(self._make_test_case(2, 8, 2, 2, 4, 8, 4, seed=1, positions=[0, 3]))

        # Zero input at position 0: the only attended value is the new (zero) V, so the
        # output must be all zeros and cache slot 0 is overwritten with zeros. Other slots
        # keep their prefilled values.
        tests.append(
            self._make_test_case(4, 16, 4, 2, 4, 16, 8, seed=2, positions=[0, 0, 0, 0], zero_x=True)
        )

        # Zero query, non-empty cache: scores are 0 on every valid slot, so attention is
        # uniform over 0..pos. Catches dropping the past cache or using a single position.
        tests.append(
            self._make_test_case(2, 8, 2, 1, 4, 16, 4, seed=9, positions=[1, 3], zero_x=True)
        )

        # GQA with 4 queries per KV head, power-of-2 sizes.
        tests.append(self._make_test_case(8, 64, 8, 2, 8, 128, 16, seed=3))

        # Multi-head attention (H_q == H_kv), larger power-of-2.
        tests.append(self._make_test_case(16, 128, 8, 8, 16, 256, 32, seed=4))

        # Non-power-of-2 head dim and FFN, mixed positions including a mid-cache slot.
        tests.append(self._make_test_case(3, 30, 5, 5, 6, 12, 7, seed=5, positions=[0, 3, 6]))

        # Non-power-of-2 batch, GQA, mid-size cache.
        tests.append(
            self._make_test_case(7, 48, 6, 3, 8, 80, 15, seed=6, positions=[1, 4, 0, 14, 7, 2, 9])
        )

        # Realistic small Llama-style decode batch.
        tests.append(self._make_test_case(8, 512, 8, 2, 64, 1408, 256, seed=7))

        # Longer cache, grouped-query, mid-size model.
        tests.append(self._make_test_case(4, 1024, 16, 4, 64, 2816, 512, seed=8))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Llama-3 3B-class decode step: 32 sequences, d_model=2048, 16 query / 4 KV heads,
        # head dim 128, SwiGLU width 5632, 2048-token KV cache. Float32 working set is ~450MB,
        # so 5 copies fit in the T4's 16GB with margin. Every sequence attends over the full
        # cache (positions = S_max - 1).
        B, d_model, H_q, H_kv, D, d_ffn, S_max = 32, 2048, 16, 4, 128, 5632, 2048
        return self._make_test_case(B, d_model, H_q, H_kv, D, d_ffn, S_max, seed=42)
