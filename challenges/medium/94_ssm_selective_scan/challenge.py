import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "SSM Selective Scan"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        skip: torch.Tensor,
        y: torch.Tensor,
        batch: int,
        seq_len: int,
        d_model: int,
        d_state: int,
    ):
        assert u.shape == (batch, seq_len, d_model)
        assert delta.shape == (batch, seq_len, d_model)
        assert A.shape == (d_model, d_state)
        assert B.shape == (batch, seq_len, d_state)
        assert C.shape == (batch, seq_len, d_state)
        assert skip.shape == (d_model,)
        assert y.shape == (batch, seq_len, d_model)
        assert (
            u.dtype == delta.dtype == A.dtype == B.dtype == C.dtype == skip.dtype == torch.float32
        )

        # Hidden state: (batch, d_model, d_state)
        h = torch.zeros(batch, d_model, d_state, device=u.device, dtype=u.dtype)

        for t in range(seq_len):
            delta_t = delta[:, t, :]  # (batch, d_model)
            u_t = u[:, t, :]  # (batch, d_model)

            # Discretize: A_bar = exp(delta_t * A)
            # delta_t: (batch, d_model) -> (batch, d_model, 1)
            # A: (d_model, d_state) -> (1, d_model, d_state)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (batch, d_model, d_state)

            # B_bar = delta_t * B_t
            # B[:, t, :]: (batch, d_state) -> (batch, 1, d_state)
            B_bar = delta_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (batch, d_model, d_state)

            # State update: h = A_bar * h + B_bar * u_t
            h = A_bar * h + B_bar * u_t.unsqueeze(-1)  # (batch, d_model, d_state)

            # Output: y_t = C_t @ h + skip * u_t
            # C[:, t, :]: (batch, d_state) -> einsum with h (batch, d_model, d_state)
            C_t = C[:, t, :]  # (batch, d_state)
            y_t = torch.einsum("bn,bdn->bd", C_t, h) + skip * u_t  # (batch, d_model)
            y[:, t, :] = y_t

    def reference_impl_jax(self, u, delta, A, B, C, skip, batch, seq_len, d_model, d_state):
        import jax
        import jax.numpy as jnp

        u = jnp.asarray(u, dtype=jnp.float32)  # (batch, seq_len, d_model)
        delta = jnp.asarray(delta, dtype=jnp.float32)  # (batch, seq_len, d_model)
        A = jnp.asarray(A, dtype=jnp.float32)  # (d_model, d_state)
        B = jnp.asarray(B, dtype=jnp.float32)  # (batch, seq_len, d_state)
        C = jnp.asarray(C, dtype=jnp.float32)  # (batch, seq_len, d_state)
        skip = jnp.asarray(skip, dtype=jnp.float32)  # (d_model,)

        batch = u.shape[0]
        d_model = u.shape[2]
        d_state = A.shape[1]

        # Move sequence axis to front for scanning.
        u_t = jnp.transpose(u, (1, 0, 2))  # (seq_len, batch, d_model)
        delta_t = jnp.transpose(delta, (1, 0, 2))  # (seq_len, batch, d_model)
        B_t = jnp.transpose(B, (1, 0, 2))  # (seq_len, batch, d_state)
        C_t = jnp.transpose(C, (1, 0, 2))  # (seq_len, batch, d_state)

        A_b = A[None, :, :]  # (1, d_model, d_state)

        def step(h, inp):
            dt, ut, bt, ct = inp  # dt,ut:(batch,d_model) bt,ct:(batch,d_state)
            A_bar = jnp.exp(dt[:, :, None] * A_b)  # (batch, d_model, d_state)
            B_bar = dt[:, :, None] * bt[:, None, :]  # (batch, d_model, d_state)
            h = A_bar * h + B_bar * ut[:, :, None]  # (batch, d_model, d_state)
            y_t = (
                jnp.einsum("bn,bdn->bd", ct, h, precision=jax.lax.Precision.HIGHEST) + skip * ut
            )  # (batch, d_model)
            return h, y_t

        h0 = jnp.zeros((batch, d_model, d_state), dtype=jnp.float32)
        _, ys = jax.lax.scan(step, h0, (delta_t, u_t, B_t, C_t))  # (seq_len, batch, d_model)
        return jnp.transpose(ys, (1, 0, 2))  # (batch, seq_len, d_model)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "u": (ctypes.POINTER(ctypes.c_float), "in"),
            "delta": (ctypes.POINTER(ctypes.c_float), "in"),
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "in"),
            "skip": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_float), "out"),
            "batch": (ctypes.c_int, "in"),
            "seq_len": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "d_state": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, batch, seq_len, d_model, d_state, zero_u=False, zero_delta=False):
        device = self.device
        dtype = torch.float32
        if zero_u:
            u = torch.zeros(batch, seq_len, d_model, device=device, dtype=dtype)
        else:
            u = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)
        if zero_delta:
            delta = torch.zeros(batch, seq_len, d_model, device=device, dtype=dtype)
        else:
            # delta must be positive
            delta = torch.rand(batch, seq_len, d_model, device=device, dtype=dtype) + 0.01
        # A must be negative for stability (eigenvalues < 0)
        A = -torch.rand(d_model, d_state, device=device, dtype=dtype) - 0.01
        B = torch.randn(batch, seq_len, d_state, device=device, dtype=dtype)
        C = torch.randn(batch, seq_len, d_state, device=device, dtype=dtype)
        skip = torch.rand(d_model, device=device, dtype=dtype)
        y = torch.empty(batch, seq_len, d_model, device=device, dtype=dtype)
        return {
            "u": u,
            "delta": delta,
            "A": A,
            "B": B,
            "C": C,
            "skip": skip,
            "y": y,
            "batch": batch,
            "seq_len": seq_len,
            "d_model": d_model,
            "d_state": d_state,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        device = self.device
        dtype = torch.float32
        batch, seq_len, d_model, d_state = 1, 4, 2, 2
        u = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]],
            device=device,
            dtype=dtype,
        )
        delta = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
        A = torch.tensor([[-0.5, -1.0], [-0.5, -1.0]], device=device, dtype=dtype)
        B = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]],
            device=device,
            dtype=dtype,
        )
        C = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]],
            device=device,
            dtype=dtype,
        )
        skip = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        y = torch.empty(batch, seq_len, d_model, device=device, dtype=dtype)
        return {
            "u": u,
            "delta": delta,
            "A": A,
            "B": B,
            "C": C,
            "skip": skip,
            "y": y,
            "batch": batch,
            "seq_len": seq_len,
            "d_model": d_model,
            "d_state": d_state,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: single token
        tests.append(self._make_test_case(1, 1, 1, 4))

        # Edge case: tiny dimensions
        tests.append(self._make_test_case(1, 2, 2, 2))

        # Edge case: zero input (output should be skip * 0 = 0)
        tests.append(self._make_test_case(1, 4, 4, 4, zero_u=True))

        # Edge case: zero delta (A_bar=1, B_bar=0, so state stays zero, output = skip * u)
        tests.append(self._make_test_case(2, 4, 4, 4, zero_delta=True))

        # Power-of-2 lengths
        tests.append(self._make_test_case(2, 16, 8, 4))
        tests.append(self._make_test_case(2, 64, 16, 8))

        # Non-power-of-2
        tests.append(self._make_test_case(2, 30, 12, 4))
        tests.append(self._make_test_case(3, 100, 24, 8))

        # Typical d_state=16 (common Mamba setting)
        tests.append(self._make_test_case(2, 128, 32, 16))

        # Realistic size
        tests.append(self._make_test_case(4, 256, 64, 16))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # batch=4, seq_len=4096, d_model=512, d_state=16
        # Memory: u+delta+y ~ 3 * 4*4096*512*4 = 96MB; A+B+C+skip small
        # Total << 1GB, comfortably fits 5x in 16GB T4
        return self._make_test_case(4, 4096, 512, 16)
