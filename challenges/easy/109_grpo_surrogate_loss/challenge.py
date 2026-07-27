import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "GRPO Surrogate Loss"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        rewards: torch.Tensor,
        log_pi: torch.Tensor,
        log_pi_old: torch.Tensor,
        log_ref: torch.Tensor,
        output: torch.Tensor,
        clip_eps: float,
        beta: float,
        B: int,
        G: int,
        S: int,
    ):
        assert rewards.shape == (B, G)
        assert log_pi.shape == (B, G, S)
        assert log_pi_old.shape == (B, G, S)
        assert log_ref.shape == (B, G, S)
        assert output.shape == (1,)
        assert (
            rewards.dtype
            == log_pi.dtype
            == log_pi_old.dtype
            == log_ref.dtype
            == output.dtype
            == torch.float32
        )

        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = ((rewards - mean_rewards) / (std_rewards + 1e-8)).unsqueeze(-1)

        ratio = torch.exp(log_pi - log_pi_old)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)

        kl_diff = log_ref - log_pi
        kl_penalty = torch.exp(kl_diff) - kl_diff - 1.0
        output[0] = -torch.mean(surrogate - beta * kl_penalty)

    def reference_impl_jax(
        self,
        rewards,
        log_pi,
        log_pi_old,
        log_ref,
        clip_eps,
        beta,
        B,
        G,
        S,
    ):
        import jax.numpy as jnp

        mean_rewards = jnp.mean(rewards, axis=1, keepdims=True)
        std_rewards = jnp.std(rewards, axis=1, keepdims=True)
        advantages = ((rewards - mean_rewards) / (std_rewards + 1e-8))[..., None]

        ratio = jnp.exp(log_pi - log_pi_old)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = jnp.minimum(ratio * advantages, clipped_ratio * advantages)

        kl_diff = log_ref - log_pi
        kl_penalty = jnp.exp(kl_diff) - kl_diff - 1.0
        return -jnp.mean(surrogate - beta * kl_penalty)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "rewards": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_pi": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_pi_old": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_ref": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "clip_eps": (ctypes.c_float, "in"),
            "beta": (ctypes.c_float, "in"),
            "B": (ctypes.c_int, "in"),
            "G": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        B,
        G,
        S,
        rewards=None,
        log_pi=None,
        log_pi_old=None,
        log_ref=None,
        clip_eps=0.2,
        beta=0.01,
    ):
        dtype = torch.float32
        device = self.device

        def tensor_or_random(values, shape):
            if values is None:
                return torch.randn(*shape, device=device, dtype=dtype)
            return torch.tensor(values, device=device, dtype=dtype)

        return {
            "rewards": tensor_or_random(rewards, (B, G)),
            "log_pi": tensor_or_random(log_pi, (B, G, S)),
            "log_pi_old": tensor_or_random(log_pi_old, (B, G, S)),
            "log_ref": tensor_or_random(log_ref, (B, G, S)),
            "output": torch.empty(1, device=device, dtype=dtype),
            "clip_eps": clip_eps,
            "beta": beta,
            "B": B,
            "G": G,
            "S": S,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._make_test_case(
            1,
            2,
            2,
            rewards=[[10.0, 0.0]],
            log_pi=[[[0.1, 0.2], [-0.5, -0.4]]],
            log_pi_old=[[[0.0, 0.0], [0.0, 0.0]]],
            log_ref=[[[0.0, 0.0], [0.0, 0.0]]],
            clip_eps=0.2,
            beta=0.01,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Hand-computed group normalization and clipping example.
        tests.append(self.generate_example_test())

        # Two groups with two responses each.
        tests.append(self._make_test_case(2, 2, 4))

        # Single response token; the advantage broadcast still has to be correct.
        tests.append(self._make_test_case(1, 2, 1, rewards=[[0.0, 2.0]]))

        # Equal rewards: advantages are zero while KL remains active.
        tests.append(
            self._make_test_case(
                2,
                4,
                8,
                rewards=[[3.0] * 4, [-2.0] * 4],
                log_pi_old=[[[0.0] * 8] * 4] * 2,
            )
        )

        # Negative and mixed rewards.
        tests.append(
            self._make_test_case(
                1,
                5,
                7,
                rewards=[[-4.0, -1.0, 0.0, 2.0, 5.0]],
                clip_eps=0.1,
            )
        )

        # Zero KL difference isolates the PPO surrogate.
        tests.append(
            self._make_test_case(
                2,
                4,
                16,
                log_ref=[[[-0.2] * 16] * 4] * 2,
                log_pi=[[[-0.2] * 16] * 4] * 2,
            )
        )

        # Power-of-two group and sequence dimensions.
        tests.append(self._make_test_case(4, 8, 64, clip_eps=0.1, beta=0.05))

        # Non-power-of-two dimensions.
        tests.append(self._make_test_case(3, 5, 27))

        # Maximum group width exercises the group reduction boundary.
        tests.append(self._make_test_case(2, 32, 3))

        # Extreme but finite KL differences.
        tests.append(
            self._make_test_case(
                2,
                4,
                8,
                log_pi=[[[8.0] * 8] * 4] * 2,
                log_ref=[[[-8.0] * 8] * 4] * 2,
            )
        )

        # Large positive, finite KL log-ratio exercises the exponential branch.
        tests.append(
            self._make_test_case(
                2,
                4,
                8,
                log_pi=[[[-8.0] * 8] * 4] * 2,
                log_ref=[[[8.0] * 8] * 4] * 2,
            )
        )

        # Realistic rollout shape.
        tests.append(self._make_test_case(8, 8, 256, beta=0.02))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        B, G, S = 64, 16, 4096
        return {
            "rewards": RandnTensor((B, G)),
            "log_pi": RandnTensor((B, G, S)),
            "log_pi_old": RandnTensor((B, G, S)),
            "log_ref": RandnTensor((B, G, S)),
            "output": OutTensor((1,)),
            "clip_eps": 0.2,
            "beta": 0.01,
            "B": B,
            "G": G,
            "S": S,
        }
