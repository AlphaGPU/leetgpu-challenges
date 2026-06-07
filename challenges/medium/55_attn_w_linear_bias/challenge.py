import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandTensor


class Challenge(ChallengeBase):
    name = "Attention with Linear Biases"
    atol = 0.0001
    rtol = 0.0001
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
        d: int,
        alpha: float,
    ):
        assert Q.shape == (M, d)
        assert K.shape == (N, d)
        assert V.shape == (N, d)
        assert output.shape == (M, d)

        scale = d**0.5
        attn = torch.matmul(Q, K.t()) / scale

        pos_bias = alpha * (
            torch.arange(M, device=Q.device).unsqueeze(1)
            - torch.arange(N, device=K.device).unsqueeze(0)
        )
        attn = attn + pos_bias

        attn = torch.softmax(attn, dim=1)  # M , N
        torch.matmul(attn, V, out=output)

    def reference_impl_jax(self, Q, K, V, M, N, d, alpha):
        import jax
        import jax.numpy as jnp

        scale = d**0.5
        attn = jnp.matmul(Q, K.T, precision=jax.lax.Precision.HIGHEST) / scale

        pos_bias = alpha * (jnp.arange(M).reshape(M, 1) - jnp.arange(N).reshape(1, N)).astype(
            attn.dtype
        )
        attn = attn + pos_bias

        attn = jax.nn.softmax(attn, axis=1)
        return jnp.matmul(attn, V, precision=jax.lax.Precision.HIGHEST)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in"),
            "alpha": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
        )
        K = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            device=self.device,
            dtype=dtype,
        )
        V = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            device=self.device,
            dtype=dtype,
        )
        output = torch.empty(2, 4, device=self.device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": 2, "N": 3, "d": 4, "alpha": 0.5}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_example 1
        tests.append(
            {
                "Q": torch.tensor([[1.0, 2.0]], device=self.device, dtype=dtype),
                "K": torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device, dtype=dtype),
                "V": torch.tensor([[3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype),
                "output": torch.empty(1, 2, device=self.device, dtype=dtype),
                "M": 1,
                "N": 2,
                "d": 2,
                "alpha": 0.8,
            }
        )

        # basic_example 2
        tests.append(
            {
                "Q": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
                ),
                "K": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.empty(2, 4, device=self.device, dtype=dtype),
                "M": 2,
                "N": 3,
                "d": 4,
                "alpha": 0.5,
            }
        )

        # zero_matrices
        tests.append(
            {
                "Q": torch.zeros((3, 5), device=self.device, dtype=dtype),
                "K": torch.zeros((3, 5), device=self.device, dtype=dtype),
                "V": torch.zeros((3, 5), device=self.device, dtype=dtype),
                "output": torch.empty(3, 5, device=self.device, dtype=dtype),
                "M": 3,
                "N": 3,
                "d": 5,
                "alpha": 0.5,
            }
        )

        # mixed_values
        tests.append(
            {
                "Q": torch.tensor(
                    [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "K": torch.tensor(
                    [[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.empty(4, 3, device=self.device, dtype=dtype),
                "M": 4,
                "N": 4,
                "d": 3,
                "alpha": 1.0,
            }
        )

        # large_matrices
        tests.append(
            {
                "Q": torch.empty((64, 32), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "output": torch.empty(64, 32, device=self.device, dtype=dtype),
                "M": 64,
                "N": 128,
                "d": 32,
                "alpha": -0.76,
            }
        )

        # different alpha
        tests.append(
            {
                "Q": torch.empty((64, 32), device=self.device, dtype=dtype).uniform_(-1, 1),
                "K": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-1, 1),
                "V": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-1, 1),
                "output": torch.empty(64, 32, device=self.device, dtype=dtype),
                "M": 64,
                "N": 128,
                "d": 32,
                "alpha": -0.3,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M, N, d = 2048, 2048, 1024
        return {
            "Q": RandTensor((M, d), -0.1, 0.1),
            "K": RandTensor((N, d), -0.1, 0.1),
            "V": RandTensor((N, d), -0.1, 0.1),
            "output": OutTensor((M, d)),
            "M": M,
            "N": N,
            "d": d,
            "alpha": 0.5,
        }
