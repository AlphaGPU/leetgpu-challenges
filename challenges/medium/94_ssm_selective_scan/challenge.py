import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="SSM Selective Scan",
            atol=1e-03,
            rtol=1e-03,
            num_gpus=1,
            access_tier="free",
        )

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
        assert u.device.type == "cuda"
        assert delta.device.type == "cuda"
        assert A.device.type == "cuda"
        assert B.device.type == "cuda"
        assert C.device.type == "cuda"
        assert skip.device.type == "cuda"
        assert y.device.type == "cuda"

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
        device = "cuda"
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
        device = "cuda"
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
