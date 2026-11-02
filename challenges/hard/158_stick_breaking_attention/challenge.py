import ctypes
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Stick-Breaking Attention"
    # The attention weights are products of sigmoids, so they lie in [0, 1] and sum to at most 1
    # per query, and every output is a sub-convex combination of V rows in [-1, 1]. The weights
    # are evaluated in log space (logsigmoid minus a running sum of softplus terms), which stays
    # finite for logits in the hundreds. Against an independent float64 implementation, the
    # float32 reference and a float32 flash-style kernel (key blocks processed from the most
    # recent backwards, carrying the running log-stick) use at most ~1% of the
    # atol + rtol * max|ref| budget on every functional case and the performance case. Softmax
    # weights, including the diagonal, an off-by-one in the product range, breaking the stick
    # from the oldest token, resetting the carry between key blocks, dropping a key tail tile,
    # truncating to a 2,048-token window or the wrong GQA head mapping miss by far more.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        B: int,
        H: int,
        H_KV: int,
        S: int,
        D: int,
    ):
        assert q.shape == (B, H, S, D), f"Expected q.shape=({B}, {H}, {S}, {D}), got {q.shape}"
        assert k.shape == (B, H_KV, S, D), f"Expected k.shape=({B}, {H_KV}, {S}, {D})"
        assert v.shape == (B, H_KV, S, D), f"Expected v.shape=({B}, {H_KV}, {S}, {D})"
        assert output.shape == (B, H, S, D), f"Expected output.shape=({B}, {H}, {S}, {D})"
        assert q.dtype == k.dtype == v.dtype == output.dtype == torch.float32
        assert H % H_KV == 0

        group = H // H_KV
        scale = 1.0 / math.sqrt(D)
        pos = torch.arange(S, device=q.device)
        # before[i, j] is True for the keys j that come strictly before query i.
        before = pos.unsqueeze(0) < pos.unsqueeze(1)
        zero_col = torch.zeros(S, 1, device=q.device, dtype=torch.float32)

        # One (sequence, query head) at a time keeps the S x S intermediates small.
        for b in range(B):
            for h in range(H):
                kv = h // group
                z = torch.matmul(q[b, h], k[b, kv].transpose(0, 1)) * scale  # (S, S)
                # log(1 - sigmoid(z)) = -softplus(z), zeroed outside the causal window.
                log_rest = torch.where(before, -F.softplus(z), torch.zeros_like(z))
                # Inclusive suffix sum over keys, then shift by one key so that column j holds
                # sum_{j < m < i} log(1 - sigmoid(z[i, m])) without any cancellation.
                suffix = torch.flip(torch.cumsum(torch.flip(log_rest, [1]), dim=1), [1])
                exclusive = torch.cat((suffix[:, 1:], zero_col), dim=1)
                log_w = F.logsigmoid(z) + exclusive
                w = torch.where(before, torch.exp(log_w), torch.zeros_like(z))
                output[b, h].copy_(torch.matmul(w, v[b, kv]))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "q": (ctypes.POINTER(ctypes.c_float), "in"),
            "k": (ctypes.POINTER(ctypes.c_float), "in"),
            "v": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "H_KV": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def _make(
        self,
        B: int,
        H: int,
        H_KV: int,
        S: int,
        D: int,
        seed: int = 0,
        q_scale: float = 1.0,
        bias: float = 0.0,
    ) -> Dict[str, Any]:
        # q, k, v are uniform in [-1, 1] (q scaled by q_scale). A negative bias is added to every
        # logit through feature 0: q[..., 0] = bias * sqrt(D) and k[..., 0] in [0.5, 1], so
        # logits are mostly negative like those of trained stick-breaking models, the sigmoids are
        # small and the stick reaches back thousands of tokens.
        # Feature 0 of v is also positive in that mode (see below).
        g = torch.Generator(device="cpu").manual_seed(seed)
        dev, f32 = self.device, torch.float32
        q = (torch.rand(B, H, S, D, generator=g) * 2 - 1) * q_scale
        k = torch.rand(B, H_KV, S, D, generator=g) * 2 - 1
        v = torch.rand(B, H_KV, S, D, generator=g) * 2 - 1
        if bias != 0.0:
            q[..., 0] = bias * math.sqrt(D)
            k[..., 0] = 0.5 + 0.5 * torch.rand(B, H_KV, S, generator=g)
            # A positive V feature, so the far history that a truncated window drops does not
            # average out.
            v[..., 0] = 0.5 + 0.5 * torch.rand(B, H_KV, S, generator=g)
        return {
            "q": q.to(device=dev, dtype=f32),
            "k": k.to(device=dev, dtype=f32),
            "v": v.to(device=dev, dtype=f32),
            "output": torch.full((B, H, S, D), 123.0, device=dev, dtype=f32),
            "B": B,
            "H": H,
            "H_KV": H_KV,
            "S": S,
            "D": D,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        return {
            "q": torch.tensor(
                [[[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]]], device=dev, dtype=f32
            ),
            "k": torch.tensor(
                [[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]]]], device=dev, dtype=f32
            ),
            "v": torch.tensor(
                [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]], device=dev, dtype=f32
            ),
            "output": torch.full((1, 1, 4, 2), 123.0, device=dev, dtype=f32),
            "B": 1,
            "H": 1,
            "H_KV": 1,
            "S": 4,
            "D": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Example from the description, checked by hand: query 0 has no earlier keys (output 0),
        # and query 3 shows the stick being broken from the most recent key backwards.
        tests.append(self.generate_example_test())

        # Single token: there is no earlier key, so the output must be all zeros (catches
        # attending to the diagonal and leaving the pre-filled output untouched).
        tests.append(self._make(1, 2, 1, 1, 8, seed=1))

        # Two tokens, GQA with two query heads sharing one KV head: query 1 sees key 0 with
        # weight sigmoid(z) only (catches softmax over one key, which gives weight 1).
        tests.append(self._make(1, 2, 1, 2, 4, seed=2))

        # Zero queries: every sigmoid is 1/2 and the weights are 1/2, 1/4, 1/8, ... from the
        # most recent key backwards (catches breaking the stick from the oldest key, and
        # including the diagonal).
        zq = self._make(2, 2, 2, 37, 16, seed=3)
        zq["q"].zero_()
        tests.append(zq)

        # Negative and mixed values with large logits (|z| well above 100): softplus / sigmoid must
        # be evaluated stably (log(1 + exp(z)) overflows float32 for z > 88).
        tests.append(self._make(1, 2, 1, 64, 32, seed=4, q_scale=100.0))

        # GQA with 8 query heads on 2 KV heads: query head h reads KV head h // 4 (catches the
        # h % H_KV mapping and reading KV head 0 for every query head).
        tests.append(self._make(1, 8, 2, 50, 16, seed=5, q_scale=4.0))

        # Batch of 3 with distinct inputs: catches reusing batch 0, or a wrong batch stride.
        tests.append(self._make(3, 4, 2, 64, 32, seed=6, q_scale=3.0, bias=-2.0))

        # Non-power-of-two everything; S = 100 leaves a 36-key tail tile for 64-key blocks.
        tests.append(self._make(2, 6, 3, 100, 40, seed=7, q_scale=4.0, bias=-3.0))

        # S = 255 with a long stick (strong negative bias): the carried log-stick crosses many
        # key blocks, catching kernels that reset it per block or drop the last partial block.
        tests.append(self._make(1, 4, 1, 255, 64, seed=8, q_scale=6.0, bias=-5.0))

        # Power-of-two shapes.
        tests.append(self._make(2, 4, 2, 512, 64, seed=9, q_scale=6.0, bias=-6.0))

        # Power-of-two, head dim 128, MQA.
        tests.append(self._make(1, 8, 1, 1024, 128, seed=10, q_scale=6.0, bias=-7.0))

        # Realistic long context: more than 10% of the stick is still left 2,048 tokens back, so
        # truncating the key range to a fixed window misses.
        tests.append(self._make(1, 4, 2, 4096, 64, seed=11, q_scale=4.0, bias=-12.0))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Prefill of 2 sequences of 8,192 tokens, 16 query heads on 4 KV heads (GQA), head dim
        # 128; the stick reaches thousands of tokens back.
        return self._make(2, 16, 4, 8192, 128, seed=158, q_scale=4.0, bias=-12.0)
