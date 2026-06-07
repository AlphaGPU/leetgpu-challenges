import ctypes
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "Causal Depthwise Conv1d"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

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

    def reference_impl_jax(self, x, weight, bias, B, L, D, K):
        import jax
        import jax.numpy as jnp

        # x (B, L, D) -> (B, D, L) for conv
        x_t = jnp.transpose(x, (0, 2, 1))  # (B, D, L)

        # Causal padding: K-1 zeros on the left.
        x_padded = jnp.pad(x_t, ((0, 0), (0, 0), (K - 1, 0)))  # (B, D, L + K - 1)

        # Depthwise (groups=D) cross-correlation with flipped kernel,
        # mirroring the torch reference. rhs shape (out=D, in/groups=1, K).
        w = jnp.flip(weight, axis=1).reshape(D, 1, K)
        result = jax.lax.conv_general_dilated(
            x_padded,
            w,
            window_strides=(1,),
            padding="VALID",
            feature_group_count=D,
            precision=jax.lax.Precision.HIGHEST,
        )  # (B, D, L)
        result = result + bias.reshape(1, D, 1)

        return jnp.transpose(result, (0, 2, 1))  # (B, L, D)

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
            device=self.device,
            dtype=torch.float32,
        )
        weight = torch.tensor(
            [[1.0, 0.0, -1.0], [1.0, 1.0, 1.0]], device=self.device, dtype=torch.float32
        )
        bias = torch.zeros(D, device=self.device, dtype=torch.float32)
        output = torch.empty(B, L, D, device=self.device, dtype=torch.float32)
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
                x = torch.tensor(x_vals, device=self.device, dtype=dtype)
            else:
                x = torch.randn(B, L, D, device=self.device, dtype=dtype)
            if w_vals is not None:
                weight = torch.tensor(w_vals, device=self.device, dtype=dtype)
            else:
                weight = torch.randn(D, K, device=self.device, dtype=dtype)
            if b_vals is not None:
                bias = torch.tensor(b_vals, device=self.device, dtype=dtype)
            else:
                bias = torch.randn(D, device=self.device, dtype=dtype)
            output = torch.empty(B, L, D, device=self.device, dtype=dtype)
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
        x_zero = torch.zeros(1, 8, 4, device=self.device, dtype=dtype)
        w_zero = torch.randn(4, 3, device=self.device, dtype=dtype)
        b_zero = torch.randn(4, device=self.device, dtype=dtype)
        test_cases.append(
            {
                "x": x_zero,
                "weight": w_zero,
                "bias": b_zero,
                "output": torch.empty(1, 8, 4, device=self.device, dtype=dtype),
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
        return {
            "x": RandnTensor((B, L, D)),
            "weight": RandnTensor((D, K)),
            "bias": RandnTensor((D,)),
            "output": OutTensor((B, L, D)),
            "B": B,
            "L": L,
            "D": D,
            "K": K,
        }
