import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "Multi-Head Cross-Attention"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        M: int,
        N: int,
        H: int,
        D: int,
    ):
        assert Q.shape == (M, H, D)
        assert K.shape == (N, H, D)
        assert V.shape == (N, H, D)
        assert output.shape == (M, H, D)
        assert Q.dtype == K.dtype == V.dtype == output.dtype == torch.float32

        # (M, H, D) -> (H, M, D); (N, H, D) -> (H, N, D)
        Qt = Q.transpose(0, 1)
        Kt = K.transpose(0, 1)
        Vt = V.transpose(0, 1)

        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(Qt, Kt.transpose(-2, -1)) * scale  # (H, M, N)
        attn = torch.softmax(scores, dim=-1)  # (H, M, N)
        out = torch.matmul(attn, Vt)  # (H, M, D)
        # (H, M, D) -> (M, H, D)
        output.copy_(out.transpose(0, 1))

    def reference_impl_jax(self, Q, K, V, M, N, H, D):
        import jax
        import jax.numpy as jnp

        Qt = jnp.transpose(Q, (1, 0, 2))
        Kt = jnp.transpose(K, (1, 0, 2))
        Vt = jnp.transpose(V, (1, 0, 2))

        scale = 1.0 / math.sqrt(D)
        scores = (
            jnp.matmul(Qt, jnp.transpose(Kt, (0, 2, 1)), precision=jax.lax.Precision.HIGHEST)
            * scale
        )
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attn, Vt, precision=jax.lax.Precision.HIGHEST)
        return jnp.transpose(out, (1, 0, 2))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N, H, D = 2, 3, 2, 2
        Q = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            device=self.device,
            dtype=dtype,
        )
        K = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            device=self.device,
            dtype=dtype,
        )
        V = torch.tensor(
            [
                [[1.0, 2.0], [7.0, 8.0]],
                [[3.0, 4.0], [9.0, 10.0]],
                [[5.0, 6.0], [11.0, 12.0]],
            ],
            device=self.device,
            dtype=dtype,
        )
        output = torch.empty((M, H, D), device=self.device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": M, "N": N, "H": H, "D": D}

    def _make_case(self, M, N, H, D, kind="randn"):
        dtype = torch.float32
        device = self.device
        if kind == "zeros":
            Q = torch.zeros((M, H, D), device=device, dtype=dtype)
            K = torch.zeros((N, H, D), device=device, dtype=dtype)
            V = torch.zeros((N, H, D), device=device, dtype=dtype)
        elif kind == "uniform":
            Q = torch.empty((M, H, D), device=device, dtype=dtype).uniform_(-1.0, 1.0)
            K = torch.empty((N, H, D), device=device, dtype=dtype).uniform_(-1.0, 1.0)
            V = torch.empty((N, H, D), device=device, dtype=dtype).uniform_(-1.0, 1.0)
        else:
            Q = torch.randn(M, H, D, device=device, dtype=dtype)
            K = torch.randn(N, H, D, device=device, dtype=dtype)
            V = torch.randn(N, H, D, device=device, dtype=dtype)
        output = torch.empty((M, H, D), device=device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": M, "N": N, "H": H, "D": D}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        dtype = torch.float32
        tests = []

        # Basic example (matches generate_example_test)
        tests.append(
            {
                "Q": torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[0.0, 1.0], [1.0, 0.0]],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "K": torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[0.0, 1.0], [1.0, 0.0]],
                        [[1.0, 1.0], [1.0, 1.0]],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [
                        [[1.0, 2.0], [7.0, 8.0]],
                        [[3.0, 4.0], [9.0, 10.0]],
                        [[5.0, 6.0], [11.0, 12.0]],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.empty((2, 2, 2), device=self.device, dtype=dtype),
                "M": 2,
                "N": 3,
                "H": 2,
                "D": 2,
            }
        )

        # Edge case: single query, single key, single head
        tests.append(self._make_case(1, 1, 1, 8))

        # Decode-like: single query vs many keys
        tests.append(self._make_case(1, 16, 4, 8))

        # Prefill-like: many queries vs single key (attention collapses to V)
        tests.append(self._make_case(4, 1, 2, 8))

        # Zero inputs (softmax should be uniform 1/N)
        tests.append(self._make_case(3, 5, 2, 4, kind="zeros"))

        # Negative + mixed values via uniform
        tests.append(self._make_case(4, 6, 2, 8, kind="uniform"))

        # Power-of-2 sizes
        tests.append(self._make_case(16, 32, 4, 32))

        # Non-power-of-2: M != N with odd dims
        tests.append(self._make_case(30, 45, 6, 32))

        # Larger non-power-of-2
        tests.append(self._make_case(100, 200, 8, 64))

        # Realistic BART/T5-like sizes
        tests.append(self._make_case(128, 512, 16, 64))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # BART-large-style cross-attention: 1024 decoder queries attending to
        # 2048 encoder tokens, 16 heads, head_dim=128.
        M, N, H, D = 1024, 2048, 16, 128
        return {
            "Q": RandnTensor((M, H, D)),
            "K": RandnTensor((N, H, D)),
            "V": RandnTensor((N, H, D)),
            "output": OutTensor((M, H, D)),
            "M": M,
            "N": N,
            "H": H,
            "D": D,
        }
