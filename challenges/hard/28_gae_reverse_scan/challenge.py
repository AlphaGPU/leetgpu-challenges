import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "Parallel Reverse Scan (GAE)"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        gamma: float,
        lam: float,
        B: int,
        S: int,
    ):
        assert rewards.shape == (B, S)
        assert values.shape == (B, S)
        assert advantages.shape == (B, S)
        assert rewards.dtype == values.dtype == advantages.dtype == torch.float32

        next_values = torch.zeros_like(values)
        if S > 1:
            next_values[:, :-1] = values[:, 1:]
        deltas = rewards + gamma * next_values - values

        last_gae = torch.zeros(B, device=self.device, dtype=rewards.dtype)
        decay = gamma * lam
        for t in reversed(range(S)):
            last_gae = deltas[:, t] + decay * last_gae
            advantages[:, t] = last_gae

    def reference_impl_jax(self, rewards, values, gamma, lam, B, S):
        import jax.numpy as jnp
        from jax import lax

        next_values = jnp.concatenate(
            [values[:, 1:], jnp.zeros((B, 1), dtype=values.dtype)], axis=1
        )
        deltas = rewards + gamma * next_values - values
        decay = gamma * lam

        def step(carry, delta_t):
            carry = delta_t + decay * carry
            return carry, carry

        _, reversed_advantages = lax.scan(
            step, jnp.zeros((B,), dtype=rewards.dtype), deltas[:, ::-1].T
        )
        return reversed_advantages.T[:, ::-1]

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "rewards": (ctypes.POINTER(ctypes.c_float), "in"),
            "values": (ctypes.POINTER(ctypes.c_float), "in"),
            "advantages": (ctypes.POINTER(ctypes.c_float), "out"),
            "gamma": (ctypes.c_float, "in"),
            "lam": (ctypes.c_float, "in"),
            "B": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, B, S, rewards=None, values=None, gamma=0.99, lam=0.95):
        dtype = torch.float32
        device = self.device
        if rewards is None:
            rewards = torch.randn(B, S, device=device, dtype=dtype)
        else:
            rewards = torch.tensor(rewards, device=device, dtype=dtype)
        if values is None:
            values = torch.randn(B, S, device=device, dtype=dtype)
        else:
            values = torch.tensor(values, device=device, dtype=dtype)
        return {
            "rewards": rewards,
            "values": values,
            "advantages": torch.empty(B, S, device=device, dtype=dtype),
            "gamma": gamma,
            "lam": lam,
            "B": B,
            "S": S,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._make_test_case(
            1, 4, rewards=[[1.0, 2.0, 3.0, 4.0]], values=[[0.5, 1.0, 1.5, 2.0]], gamma=0.9, lam=0.5,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        tests.append(self._make_test_case(1, 1, rewards=[[2.0]], values=[[0.5]]))
        tests.append(self._make_test_case(1, 2, rewards=[[1.0, -1.0]], values=[[0.0, 0.5]]))
        tests.append(
            self._make_test_case(2, 4, rewards=[[0.0] * 4, [0.0] * 4], values=[[0.0] * 4] * 2)
        )
        tests.append(
            self._make_test_case(
                1, 4, rewards=[[-1.0, -2.0, 3.0, -4.0]], values=[[1.0, -1.0, 2.0, -2.0]],
            )
        )
        tests.append(self._make_test_case(4, 16))
        tests.append(self._make_test_case(8, 64, gamma=1.0, lam=1.0))
        tests.append(self._make_test_case(2, 30))
        tests.append(self._make_test_case(4, 100, gamma=0.9, lam=0.8))
        tests.append(self._make_test_case(16, 1024))
        tests.append(self._make_test_case(64, 4096))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        B, S = 64, 4096
        return {
            "rewards": RandnTensor((B, S)),
            "values": RandnTensor((B, S)),
            "advantages": OutTensor((B, S)),
            "gamma": 0.99,
            "lam": 0.95,
            "B": B,
            "S": S,
        }
