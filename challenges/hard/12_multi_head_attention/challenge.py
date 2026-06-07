import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandTensor


class Challenge(ChallengeBase):
    name = "Multi-Head Attention"
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
        N: int,
        d_model: int,
        h: int,
    ):
        assert Q.shape == (N, d_model)
        assert K.shape == (N, d_model)
        assert V.shape == (N, d_model)
        assert output.shape == (N, d_model)
        assert Q.dtype == K.dtype == V.dtype == output.dtype
        assert Q.device == K.device == V.device == output.device
        d_k = d_model // h
        result = torch.zeros((N, d_model), dtype=Q.dtype, device=Q.device)
        for head in range(h):
            Q_h = Q[:, head * d_k : (head + 1) * d_k]
            K_h = K[:, head * d_k : (head + 1) * d_k]
            V_h = V[:, head * d_k : (head + 1) * d_k]
            scores = torch.matmul(Q_h, K_h.t()) / (d_k**0.5)
            softmax = torch.softmax(scores, dim=1)
            head_output = torch.matmul(softmax, V_h)
            result[:, head * d_k : (head + 1) * d_k] = head_output
        output.copy_(result)

    def reference_impl_jax(self, Q, K, V, N, d_model, h):
        import jax
        import jax.numpy as jnp

        d_k = d_model // h
        # Reshape (N, d_model) -> (N, h, d_k) -> (h, N, d_k)
        Q_h = jnp.transpose(jnp.reshape(Q, (N, h, d_k)), (1, 0, 2))
        K_h = jnp.transpose(jnp.reshape(K, (N, h, d_k)), (1, 0, 2))
        V_h = jnp.transpose(jnp.reshape(V, (N, h, d_k)), (1, 0, 2))
        scores = jnp.matmul(Q_h, jnp.transpose(K_h, (0, 2, 1))) / (d_k**0.5)
        softmax = jax.nn.softmax(scores, axis=-1)
        head_output = jnp.matmul(softmax, V_h)  # (h, N, d_k)
        # (h, N, d_k) -> (N, h, d_k) -> (N, d_model)
        result = jnp.reshape(jnp.transpose(head_output, (1, 0, 2)), (N, d_model))
        return result

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "h": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor(
            [[1.0, 0.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], device=self.device, dtype=dtype
        )
        K = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device=self.device, dtype=dtype
        )
        V = torch.tensor(
            [[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]], device=self.device, dtype=dtype
        )
        output = torch.empty(2, 4, device=self.device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "N": 2,
            "d_model": 4,
            "h": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []
        # basic_example
        Q = torch.tensor(
            [[1.0, 0.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], device=self.device, dtype=dtype
        )
        K = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device=self.device, dtype=dtype
        )
        V = torch.tensor(
            [[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]], device=self.device, dtype=dtype
        )
        output = torch.empty(2, 4, device=self.device, dtype=dtype)
        test_cases.append({"Q": Q, "K": K, "V": V, "output": output, "N": 2, "d_model": 4, "h": 2})
        # single_head
        Q = torch.tensor([[1.0, 1.0], [2.0, 2.0]], device=self.device, dtype=dtype)
        K = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=self.device, dtype=dtype)
        V = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device=self.device, dtype=dtype)
        output = torch.empty(2, 2, device=self.device, dtype=dtype)
        test_cases.append({"Q": Q, "K": K, "V": V, "output": output, "N": 2, "d_model": 2, "h": 1})
        # four_heads (random)
        Q = torch.empty(4, 4, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        K = torch.empty(4, 4, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        V = torch.empty(4, 4, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        output = torch.empty(4, 4, device=self.device, dtype=dtype)
        test_cases.append({"Q": Q, "K": K, "V": V, "output": output, "N": 4, "d_model": 4, "h": 4})
        # medium_size (random)
        Q = torch.empty(32, 32, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        K = torch.empty(32, 32, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        V = torch.empty(32, 32, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        output = torch.empty(32, 32, device=self.device, dtype=dtype)
        test_cases.append(
            {"Q": Q, "K": K, "V": V, "output": output, "N": 32, "d_model": 32, "h": 8}
        )
        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        return {
            "Q": RandTensor((1024, 1024), -10.0, 10.0),
            "K": RandTensor((1024, 1024), -10.0, 10.0),
            "V": RandTensor((1024, 1024), -10.0, 10.0),
            "output": OutTensor((1024, 1024)),
            "N": 1024,
            "d_model": 1024,
            "h": 16,
        }
