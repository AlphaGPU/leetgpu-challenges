import ctypes
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Causal Depthwise Conv1d",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
        B: int,
        L: int,
        D: int,
        K: int,
    ):
        assert x.shape == (B, L, D)
        assert weight.shape == (D, K)
        assert bias.shape == (D,)
        assert output.shape == (B, L, D)
        assert x.dtype == weight.dtype == bias.dtype == output.dtype == torch.float32
        assert x.device.type == "cuda"
        assert weight.device.type == "cuda"
        assert bias.device.type == "cuda"
        assert output.device.type == "cuda"

        # Reshape to (B, D, L) for conv1d
        x_t = x.permute(0, 2, 1).contiguous()  # (B, D, L)

        # Causal padding: pad K-1 zeros on the left so each output position
        # only sees current and past input positions
        x_padded = F.pad(x_t, (K - 1, 0))  # (B, D, L + K - 1)

        # Depthwise conv: weight (D, K) -> (D, 1, K), groups=D
        # Flip the kernel so weight[d, 0] applies to the current position (l-0)
        # and weight[d, K-1] applies to the oldest position (l-(K-1)).
        # F.conv1d uses cross-correlation (no implicit flip), so we flip explicitly.
        w = weight.flip(1).unsqueeze(1)  # (D, 1, K)
        result = F.conv1d(x_padded, w, bias=bias, groups=D)  # (B, D, L)

        output.copy_(result.permute(0, 2, 1))  # (B, L, D)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "bias": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "L": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        B, L, D, K = 1, 4, 2, 3
        x = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
            device="cuda",
            dtype=torch.float32,
        )
        weight = torch.tensor(
            [[1.0, 0.0, -1.0], [1.0, 1.0, 1.0]], device="cuda", dtype=torch.float32
        )
        bias = torch.zeros(D, device="cuda", dtype=torch.float32)
        output = torch.empty(B, L, D, device="cuda", dtype=torch.float32)
        return {
            "x": x,
            "weight": weight,
            "bias": bias,
            "output": output,
            "B": B,
            "L": L,
            "D": D,
            "K": K,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []

        def make_case(B, L, D, K, x_vals=None, w_vals=None, b_vals=None):
            if x_vals is not None:
                x = torch.tensor(x_vals, device="cuda", dtype=dtype)
            else:
                x = torch.randn(B, L, D, device="cuda", dtype=dtype)
            if w_vals is not None:
                weight = torch.tensor(w_vals, device="cuda", dtype=dtype)
            else:
                weight = torch.randn(D, K, device="cuda", dtype=dtype)
            if b_vals is not None:
                bias = torch.tensor(b_vals, device="cuda", dtype=dtype)
            else:
                bias = torch.randn(D, device="cuda", dtype=dtype)
            output = torch.empty(B, L, D, device="cuda", dtype=dtype)
            return {
                "x": x,
                "weight": weight,
                "bias": bias,
                "output": output,
                "B": B,
                "L": L,
                "D": D,
                "K": K,
            }

        # Example test (matches generate_example_test)
        test_cases.append(
            make_case(
                1,
                4,
                2,
                3,
                x_vals=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
                w_vals=[[1.0, 0.0, -1.0], [1.0, 1.0, 1.0]],
                b_vals=[0.0, 0.0],
            )
        )

        # Edge cases: minimal sizes
        test_cases.append(make_case(1, 1, 1, 1))  # single element, kernel=1
        test_cases.append(make_case(1, 2, 1, 2))  # L < K, so first output is partial
        test_cases.append(make_case(2, 3, 4, 3))  # small batch, B=2

        # Zero inputs
        x_zero = torch.zeros(1, 8, 4, device="cuda", dtype=dtype)
        w_zero = torch.randn(4, 3, device="cuda", dtype=dtype)
        b_zero = torch.randn(4, device="cuda", dtype=dtype)
        test_cases.append(
            {
                "x": x_zero,
                "weight": w_zero,
                "bias": b_zero,
                "output": torch.empty(1, 8, 4, device="cuda", dtype=dtype),
                "B": 1,
                "L": 8,
                "D": 4,
                "K": 3,
            }
        )

        # Negative values
        test_cases.append(make_case(1, 16, 8, 4))

        # Power-of-2 sizes
        test_cases.append(make_case(2, 32, 16, 4))
        test_cases.append(make_case(4, 64, 32, 4))

        # Non-power-of-2 sizes
        test_cases.append(make_case(3, 30, 12, 3))
        test_cases.append(make_case(2, 100, 24, 4))

        # Realistic inference size (Mamba-like small)
        test_cases.append(make_case(2, 256, 128, 4))

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        B, L, D, K = 8, 2048, 4096, 4
        dtype = torch.float32
        x = torch.randn(B, L, D, device="cuda", dtype=dtype)
        weight = torch.randn(D, K, device="cuda", dtype=dtype)
        bias = torch.randn(D, device="cuda", dtype=dtype)
        output = torch.empty(B, L, D, device="cuda", dtype=dtype)
        return {
            "x": x,
            "weight": weight,
            "bias": bias,
            "output": output,
            "B": B,
            "L": L,
            "D": D,
            "K": K,
        }
