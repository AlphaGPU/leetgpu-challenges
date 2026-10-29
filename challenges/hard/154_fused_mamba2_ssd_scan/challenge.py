import ctypes
import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


def _segsum(x: torch.Tensor) -> torch.Tensor:
    # x: (..., T) -> (..., T, T) with out[..., i, j] = sum_{j < k <= i} x[..., k] for j <= i and
    # -inf above the diagonal. Built from a masked cumsum instead of cumsum differences, so a
    # long run of decays does not lose precision to cancellation.
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    strict = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~strict, 0.0)
    out = torch.cumsum(x, dim=-2)
    lower = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return out.masked_fill(~lower, float("-inf"))


class Challenge(ChallengeBase):
    name = "Fused Mamba-2 SSD Scan"
    # Every output is (y + D * x) * silu(z) where y is a decayed sum of rank-1 updates whose
    # weights are all positive and at most 1 (exp of a non-positive log-decay), so float32
    # rounding error grows with the effective memory length 1 / (delta * |A|), not with seq_len.
    # Against an independent float64 sequential recurrence the float32 chunked reference stays
    # well inside atol + rtol * max|ref| on every functional case and the performance case.
    # Dropping the softplus or dt_bias, skipping the D term or the gate, using sigmoid instead of
    # silu, h % num_groups mapping, a missing delta on the input, decaying after the update,
    # resetting the state at chunk boundaries or leaking it across the batch all miss by orders
    # of magnitude more (see the targeted cases below).
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        dt_bias: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        state_dim: int,
    ):
        b, L, H, P, G, N = batch, seq_len, num_heads, head_dim, num_groups, state_dim
        assert x.shape == (b, L, H, P)
        assert z.shape == (b, L, H, P)
        assert y.shape == (b, L, H, P)
        assert dt.shape == (b, L, H)
        assert dt_bias.shape == (H,)
        assert A.shape == (H,)
        assert D.shape == (H,)
        assert B.shape == (b, L, G, N)
        assert C.shape == (b, L, G, N)
        for t in (x, dt, dt_bias, A, B, C, D, z, y):
            assert t.dtype == torch.float32
        assert H % G == 0
        heads_per_group = H // G

        # Discretization: delta = softplus(dt + dt_bias), per-step log-decay delta * A (<= 0),
        # and the input enters the state scaled by delta.
        delta = F.softplus(dt + dt_bias)  # (b, L, H)
        log_decay = delta * A  # (b, L, H)
        xd = x * delta.unsqueeze(-1)  # (b, L, H, P)

        # Chunked state space duality (SSD) form of the recurrence
        #   S_t = exp(delta_t * A) * S_{t-1} + (delta_t * x_t) B_t^T,   y_t = S_t C_t
        # with the sequence split into chunks of T: a masked quadratic form inside each chunk
        # plus the chunk-final states carried across chunks.
        T = 64
        pad = (-L) % T
        if pad:
            log_decay = F.pad(log_decay, (0, 0, 0, pad))
            xd = F.pad(xd, (0, 0, 0, 0, 0, pad))
            Bp = F.pad(B, (0, 0, 0, 0, 0, pad))
            Cp = F.pad(C, (0, 0, 0, 0, 0, pad))
        else:
            Bp, Cp = B, C
        nc = (L + pad) // T

        xd = xd.reshape(b, nc, T, H, P)
        Bc = Bp.reshape(b, nc, T, G, N)
        Cc = Cp.reshape(b, nc, T, G, N)
        a = log_decay.reshape(b, nc, T, H).permute(0, 3, 1, 2)  # (b, H, nc, T)
        a_cs = torch.cumsum(a, dim=-1)

        # 1. intra-chunk: y[l] += sum_{s <= l} (C_l . B_s) exp(sum_{s < k <= l} a_k) xd_s
        decay_ls = torch.exp(_segsum(a))  # (b, H, nc, T, T)
        cb = torch.einsum("bclgn,bcsgn->bcgls", Cc, Bc)  # (b, nc, G, T, T)
        cb = cb.repeat_interleave(heads_per_group, dim=2)  # (b, nc, H, T, T)
        w = cb * decay_ls.permute(0, 2, 1, 3, 4)
        y_diag = torch.einsum("bchls,bcshp->bclhp", w, xd)

        # 2. state at the end of each chunk from that chunk's inputs only
        decay_to_end = torch.exp(a_cs[..., -1:] - a_cs)  # (b, H, nc, T)
        xw = xd * decay_to_end.permute(0, 2, 3, 1).unsqueeze(-1)  # (b, nc, T, H, P)
        xw = xw.reshape(b, nc, T, G, heads_per_group, P)
        states = torch.einsum("bctgjp,bctgn->bcgjpn", xw, Bc).reshape(b, nc, H, P, N)

        # 3. carry the states across chunks (state before chunk 0 is zero for every sequence)
        states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
        chunk_decay = torch.exp(_segsum(F.pad(a_cs[..., -1], (1, 0))))  # (b, H, nc+1, nc+1)
        states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)[:, :-1]

        # 4. contribution of the carried-in state to every position of the chunk
        decay_from_start = torch.exp(a_cs).permute(0, 2, 3, 1)  # (b, nc, T, H)
        states_g = states.reshape(b, nc, G, heads_per_group, P, N)
        y_off = torch.einsum("bclgn,bcgjpn->bclgjp", Cc, states_g).reshape(b, nc, T, H, P)
        y_off = y_off * decay_from_start.unsqueeze(-1)

        ssm = (y_diag + y_off).reshape(b, nc * T, H, P)[:, :L]
        out = (ssm + D.view(1, 1, H, 1) * x) * F.silu(z)
        y.copy_(out)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "dt": (ctypes.POINTER(ctypes.c_float), "in"),
            "dt_bias": (ctypes.POINTER(ctypes.c_float), "in"),
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "in"),
            "D": (ctypes.POINTER(ctypes.c_float), "in"),
            "z": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_float), "out"),
            "batch": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "num_heads": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
            "num_groups": (ctypes.c_int, "in"),
            "state_dim": (ctypes.c_int, "in"),
        }

    def _t(self, data) -> torch.Tensor:
        return torch.tensor(data, device=self.device, dtype=torch.float32)

    def _rand(self, shape, low: float, high: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(low, high)

    def _case(self, x, dt, dt_bias, A, B, C, D, z) -> Dict[str, Any]:
        batch, seq_len, num_heads, head_dim = x.shape
        num_groups, state_dim = B.shape[2], B.shape[3]
        return {
            "x": x,
            "dt": dt,
            "dt_bias": dt_bias,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "z": z,
            # Pre-filled so a kernel that skips heads, positions or a tile tail fails instead of
            # passing on leftover zeros.
            "y": torch.full(x.shape, 123.0, device=self.device, dtype=torch.float32),
            "batch": batch,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_groups": num_groups,
            "state_dim": state_dim,
        }

    def _rand_case(
        self,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        state_dim: int,
        seed: int,
        a_range=(1.0, 16.0),
        dt_bias_range=(-4.0, -1.0),
        dt_range=(-2.0, 2.0),
    ) -> Dict[str, Any]:
        # Mamba-2 style ranges by default: A = -exp(uniform(log 1, log 16)) and
        # softplus(dt + dt_bias) mostly between 0.002 and 0.7, so per-step decays
        # exp(delta * A) range from about 0.998 down to near 0.
        torch.manual_seed(seed)
        b, L, H, P, G, N = batch, seq_len, num_heads, head_dim, num_groups, state_dim
        A = -torch.exp(self._rand((H,), math.log(a_range[0]), math.log(a_range[1])))
        return self._case(
            self._rand((b, L, H, P), -1.0, 1.0),
            self._rand((b, L, H), *dt_range),
            self._rand((H,), *dt_bias_range),
            A,
            self._rand((b, L, G, N), -1.0, 1.0),
            self._rand((b, L, G, N), -1.0, 1.0),
            self._rand((H,), -1.0, 1.0),
            self._rand((b, L, H, P), -3.0, 3.0),
        )

    def generate_example_test(self) -> Dict[str, Any]:
        # batch = 1, seq_len = 3, one head with head_dim = 2, one group with state_dim = 2.
        # dt + dt_bias = 0 gives delta = softplus(0) = ln 2, and A = -1 then gives a per-step
        # decay of exp(-ln 2) = 0.5.
        return self._case(
            x=self._t([[[[1.0, 2.0]], [[0.0, -1.0]], [[2.0, 0.0]]]]),
            dt=self._t([[[0.0], [0.0], [0.0]]]),
            dt_bias=self._t([0.0]),
            A=self._t([-1.0]),
            B=self._t([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]]),
            C=self._t([[[[1.0, 0.0]], [[1.0, 1.0]], [[0.0, 1.0]]]]),
            D=self._t([0.5]),
            z=self._t([[[[1.0, 1.0]], [[1.0, 1.0]], [[1.0, 1.0]]]]),
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests: List[Dict[str, Any]] = []

        # 1: basic example (same as generate_example_test), checked by hand in the description
        tests.append(self.generate_example_test())

        # 2: single token: the state starts at zero, so
        # y = (softplus(dt + dt_bias) * (C . B) * x + D * x) * silu(z). Negative dt, negative and
        # positive gates; catches a missing softplus, a missing dt_bias, a missing D term and
        # sigmoid(z) in place of silu(z).
        tests.append(
            self._case(
                x=self._t([[[[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]]]]),
                dt=self._t([[[-1.5, 0.75]]]),
                dt_bias=self._t([0.5, -0.25]),
                A=self._t([-2.0, -0.5]),
                B=self._t([[[[0.5, -1.0, 2.0, 1.0]]]]),
                C=self._t([[[[1.0, 1.0, -0.5, 2.0]]]]),
                D=self._t([0.25, -1.0]),
                z=self._t([[[[2.0, -1.0, 0.0], [-3.0, 1.5, 4.0]]]]),
            )
        )

        # 3: all-zero inputs (A = -1, D = 0): output is exactly zero
        tests.append(
            self._case(
                x=torch.zeros((1, 16, 2, 4), device=self.device),
                dt=torch.zeros((1, 16, 2), device=self.device),
                dt_bias=torch.zeros((2,), device=self.device),
                A=-torch.ones((2,), device=self.device),
                B=torch.zeros((1, 16, 1, 8), device=self.device),
                C=torch.zeros((1, 16, 1, 8), device=self.device),
                D=torch.zeros((2,), device=self.device),
                z=torch.zeros((1, 16, 2, 4), device=self.device),
            )
        )

        # 4: long memory (|A| <= 0.01, delta about 0.5 - 2.1, decay >= 0.98 per step) over 300
        # steps: the state must be carried across any chunk or tile boundary. Resetting it per
        # chunk, or decaying with exp(A) instead of exp(delta * A), misses badly.
        tests.append(
            self._rand_case(
                1, 300, 2, 8, 1, 16, seed=4, a_range=(0.001, 0.01), dt_bias_range=(0.0, 1.0)
            )
        )

        # 5: short memory (A <= -8, delta large, decay below 1e-17): y_t is only the current
        # token's own update, so decaying the state after adding the input
        # (S_t = a_t * (S_{t-1} + delta_t x_t B_t^T)) gives about 0 instead. dt + dt_bias reaches
        # 25, past the linear switch-over of common softplus implementations.
        tests.append(
            self._rand_case(
                1,
                64,
                4,
                16,
                2,
                16,
                seed=5,
                a_range=(8.0, 16.0),
                dt_bias_range=(4.0, 5.0),
                dt_range=(0.0, 20.0),
            )
        )

        # 6: grouped B / C: head h reads group h // (num_heads / num_groups), so heads 0-3 use
        # group 0 and heads 4-7 group 1. h % num_groups mapping, or always group 0, differs.
        tests.append(self._rand_case(1, 64, 8, 16, 2, 32, seed=6))

        # 7: batch of 3 independent sequences with one head per group: the state restarts at
        # zero for every sequence. Treating the batch as one long sequence leaks the state.
        tests.append(
            self._rand_case(
                3, 100, 4, 8, 4, 16, seed=7, a_range=(0.01, 0.1), dt_bias_range=(0.0, 1.0)
            )
        )

        # 8: everything non-power-of-two: tails in the sequence (255), head_dim (34) and
        # state_dim (17), three groups of two heads, batch 2
        tests.append(self._rand_case(2, 255, 6, 34, 3, 17, seed=8))

        # 9: long sequence with tile tails, one group shared by all heads, head_dim = 64,
        # state_dim = 128
        tests.append(self._rand_case(1, 1000, 4, 64, 1, 128, seed=9))

        # 10: realistic size: performance-test head layout, shorter sequence
        tests.append(self._rand_case(1, 2048, 32, 64, 4, 128, seed=10))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        batch, seq_len, num_heads, head_dim, num_groups, state_dim = 2, 4096, 32, 64, 4, 128
        torch.manual_seed(154)
        return {
            "x": RandnTensor((batch, seq_len, num_heads, head_dim)),
            "dt": RandnTensor((batch, seq_len, num_heads)),
            "dt_bias": self._rand((num_heads,), -4.0, -1.0),
            "A": -torch.exp(self._rand((num_heads,), 0.0, math.log(16.0))),
            "B": RandnTensor((batch, seq_len, num_groups, state_dim)),
            "C": RandnTensor((batch, seq_len, num_groups, state_dim)),
            "D": self._rand((num_heads,), -1.0, 1.0),
            "z": RandnTensor((batch, seq_len, num_heads, head_dim)),
            "y": OutTensor((batch, seq_len, num_heads, head_dim)),
            "batch": batch,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_groups": num_groups,
            "state_dim": state_dim,
        }
