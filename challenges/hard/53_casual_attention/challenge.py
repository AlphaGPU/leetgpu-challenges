import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandTensor


class Challenge(ChallengeBase):
    name = "Causal Self-Attention"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        M: int,
        d: int,
    ):
        scale = d**0.5
        attn = torch.matmul(Q, K.t()) / scale

        # add mask
        mask = torch.triu(torch.ones(M, M, device=attn.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn, dim=1)
        torch.matmul(attn, V, out=output)

    def reference_impl_jax(self, Q, K, V, M, d):
        import jax
        import jax.numpy as jnp

        scale = d**0.5
        attn = jnp.matmul(Q, K.T) / scale

        mask = jnp.triu(jnp.ones((M, M), dtype=bool), k=1)
        attn = jnp.where(mask, -jnp.inf, attn)
        attn = jax.nn.softmax(attn, axis=1)
        return jnp.matmul(attn, V)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
        )
        K = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
        )
        V = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device=self.device, dtype=dtype
        )
        output = torch.empty(2, 4, device=self.device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": 2, "d": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_example
        tests.append(
            {
                "Q": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
                ),
                "K": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
                ),
                "V": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device=self.device, dtype=dtype
                ),
                "output": torch.empty(2, 4, device=self.device, dtype=dtype),
                "M": 2,
                "d": 4,
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
                "d": 5,
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
                "d": 3,
            }
        )

        # large_matrices
        tests.append(
            {
                "Q": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty((128, 32), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "output": torch.empty(128, 32, device=self.device, dtype=dtype),
                "M": 128,
                "d": 32,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M, d = 5000, 128
        return {
            "Q": RandTensor((M, d), -100.0, 100.0),
            "K": RandTensor((M, d), -100.0, 100.0),
            "V": RandTensor((M, d), -100.0, 100.0),
            "output": OutTensor((M, d)),
            "M": M,
            "d": d,
        }
