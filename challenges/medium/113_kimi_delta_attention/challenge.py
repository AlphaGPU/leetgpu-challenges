import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Kimi Delta Attention"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        output: torch.Tensor,
        batch: int,
        seq_len: int,
        d: int,
    ):
        assert Q.shape == (batch, seq_len, d)
        assert K.shape == (batch, seq_len, d)
        assert V.shape == (batch, seq_len, d)
        assert alpha.shape == (batch, seq_len, d)
        assert beta.shape == (batch, seq_len)
        assert output.shape == (batch, seq_len, d)
        assert (
            Q.dtype
            == K.dtype
            == V.dtype
            == alpha.dtype
            == beta.dtype
            == output.dtype
            == torch.float32
        )

        # State S: (batch, d, d) associative memory mapping keys to values.
        S = torch.zeros(batch, d, d, device=Q.device, dtype=Q.dtype)

        # Kept it sequential; there may be better optimizations in the torch form
        for t in range(seq_len):
            k_t = K[:, t, :]  # (batch, d)
            v_t = V[:, t, :]  # (batch, d)
            a_t = alpha[:, t, :]  # (batch, d)
            b_t = beta[:, t]  # (batch,)

            # Channel-wise decay: S <- Diag(a_t) S
            S = S * a_t.unsqueeze(-1)

            # Delta rule: S <- (I - b_t k_t k_t^T) S + b_t k_t v_t^T
            kS = torch.einsum("bi,bij->bj", k_t, S)  # (batch, d) = k_t^T S
            S = S + b_t[:, None, None] * k_t.unsqueeze(-1) * (v_t - kS).unsqueeze(1)

            # Output: o_t = S^T q_t
            output[:, t, :] = torch.einsum("bi,bij->bj", Q[:, t, :], S)

    def reference_impl_jax(self, Q, K, V, alpha, beta, batch, seq_len, d):
        import jax
        import jax.numpy as jnp

        Q = jnp.asarray(Q, dtype=jnp.float32)  # (batch, seq_len, d)
        K = jnp.asarray(K, dtype=jnp.float32)  # (batch, seq_len, d)
        V = jnp.asarray(V, dtype=jnp.float32)  # (batch, seq_len, d)
        alpha = jnp.asarray(alpha, dtype=jnp.float32)  # (batch, seq_len, d)
        beta = jnp.asarray(beta, dtype=jnp.float32)  # (batch, seq_len)

        batch = Q.shape[0]
        d = Q.shape[2]

        # Move sequence axis to front for scanning.
        Q_t = jnp.transpose(Q, (1, 0, 2))  # (seq_len, batch, d)
        K_t = jnp.transpose(K, (1, 0, 2))  # (seq_len, batch, d)
        V_t = jnp.transpose(V, (1, 0, 2))  # (seq_len, batch, d)
        a_t = jnp.transpose(alpha, (1, 0, 2))  # (seq_len, batch, d)
        b_t = jnp.transpose(beta, (1, 0))  # (seq_len, batch)

        def step(S, inp):
            qt, kt, vt, at, bt = inp
            S = S * at[:, :, None]
            kS = jnp.einsum("bi,bij->bj", kt, S, precision=jax.lax.Precision.HIGHEST)
            S = S + bt[:, None, None] * kt[:, :, None] * (vt - kS)[:, None, :]
            ot = jnp.einsum("bi,bij->bj", qt, S, precision=jax.lax.Precision.HIGHEST)
            return S, ot

        S0 = jnp.zeros((batch, d, d), dtype=jnp.float32)
        _, outs = jax.lax.scan(step, S0, (Q_t, K_t, V_t, a_t, b_t))  # (seq_len, batch, d)
        return jnp.transpose(outs, (1, 0, 2))  # (batch, seq_len, d)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "alpha": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "batch": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, batch, seq_len, d, zero_kv=False, alpha_one=False):
        device = self.device
        dtype = torch.float32
        Q = torch.randn(batch, seq_len, d, device=device, dtype=dtype)
        if zero_kv:
            K = torch.zeros(batch, seq_len, d, device=device, dtype=dtype)
            V = torch.zeros(batch, seq_len, d, device=device, dtype=dtype)
        else:
            # Keys are L2-normalized so (I - beta k k^T) is contractive.
            K = torch.randn(batch, seq_len, d, device=device, dtype=dtype)
            K = K / K.norm(dim=-1, keepdim=True)
            V = torch.randn(batch, seq_len, d, device=device, dtype=dtype)
        if alpha_one:
            alpha = torch.ones(batch, seq_len, d, device=device, dtype=dtype)
        else:
            # Channel-wise decay in (0.5, 1.0)
            alpha = 0.5 + 0.5 * torch.rand(batch, seq_len, d, device=device, dtype=dtype)
        beta = torch.rand(batch, seq_len, device=device, dtype=dtype)
        output = torch.empty(batch, seq_len, d, device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "alpha": alpha,
            "beta": beta,
            "output": output,
            "batch": batch,
            "seq_len": seq_len,
            "d": d,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        Q = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]], device=device, dtype=dtype)
        K = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device, dtype=dtype)
        V = torch.tensor([[[1.0, 2.0], [2.0, 0.0]]], device=device, dtype=dtype)
        alpha = torch.tensor([[[1.0, 1.0], [0.5, 0.5]]], device=device, dtype=dtype)
        beta = torch.tensor([[1.0, 0.5]], device=device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "alpha": alpha,
            "beta": beta,
            "output": torch.empty(1, 2, 2, device=device, dtype=dtype),
            "batch": 1,
            "seq_len": 2,
            "d": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # minimal edge case: single token, single channel
        tests.append(self._make_test_case(1, 1, 1))

        # tiny dimensions
        tests.append(self._make_test_case(1, 2, 2))

        # zero keys and values: state stays zero, output must be zero
        tests.append(self._make_test_case(1, 4, 4, zero_kv=True))

        # alpha = 1: pure DeltaNet recurrence with no decay
        tests.append(self._make_test_case(2, 8, 4, alpha_one=True))

        # power-of-2 sizes
        tests.append(self._make_test_case(2, 16, 8))
        tests.append(self._make_test_case(2, 64, 32))

        # non-power-of-2 sizes
        tests.append(self._make_test_case(1, 30, 7))
        tests.append(self._make_test_case(3, 100, 24))

        # realistic: chunk-boundary-crossing length with typical head dim
        tests.append(self._make_test_case(2, 200, 64))

        # realistic: full head dim
        tests.append(self._make_test_case(2, 256, 128))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # batch=4, seq_len=8192, d=128
        # Q+K+V+alpha+output ~ 5 * 4*8192*128*4 = 80MB; beta small
        # Total << 1GB, comfortably fits 5x in 16GB T4
        return self._make_test_case(4, 8192, 128)
