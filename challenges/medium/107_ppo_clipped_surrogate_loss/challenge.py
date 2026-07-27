import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "PPO Clipped Surrogate Loss"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        advantages: torch.Tensor,
        log_pi: torch.Tensor,
        log_pi_old: torch.Tensor,
        log_ref: torch.Tensor,
        output: torch.Tensor,
        clip_eps: float,
        beta: float,
        B: int,
        S: int,
    ):
        assert advantages.shape == (B, S)
        assert log_pi.shape == (B, S)
        assert log_pi_old.shape == (B, S)
        assert log_ref.shape == (B, S)
        assert output.shape == (1,)
        assert (
            advantages.dtype
            == log_pi.dtype
            == log_pi_old.dtype
            == log_ref.dtype
            == output.dtype
            == torch.float32
        )

        ratio = torch.exp(log_pi - log_pi_old)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        kl_diff = log_ref - log_pi
        kl_penalty = torch.exp(kl_diff) - kl_diff - 1.0
        output[0] = -torch.mean(surrogate - beta * kl_penalty)

    def reference_impl_jax(
        self,
        advantages,
        log_pi,
        log_pi_old,
        log_ref,
        clip_eps,
        beta,
        B,
        S,
    ):
        import jax.numpy as jnp

        ratio = jnp.exp(log_pi - log_pi_old)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = jnp.minimum(ratio * advantages, clipped_ratio * advantages)
        kl_diff = log_ref - log_pi
        kl_penalty = jnp.exp(kl_diff) - kl_diff - 1.0
        return -jnp.mean(surrogate - beta * kl_penalty)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "advantages": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_pi": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_pi_old": (ctypes.POINTER(ctypes.c_float), "in"),
            "log_ref": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "clip_eps": (ctypes.c_float, "in"),
            "beta": (ctypes.c_float, "in"),
            "B": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        B,
        S,
        advantages=None,
        log_pi=None,
        log_pi_old=None,
        log_ref=None,
        clip_eps=0.2,
        beta=0.01,
    ):
        dtype = torch.float32
        device = self.device

        def tensor_or_random(values):
            if values is None:
                return torch.randn(B, S, device=device, dtype=dtype)
            return torch.tensor(values, device=device, dtype=dtype)

        return {
            "advantages": tensor_or_random(advantages),
            "log_pi": tensor_or_random(log_pi),
            "log_pi_old": tensor_or_random(log_pi_old),
            "log_ref": tensor_or_random(log_ref),
            "output": torch.empty(1, device=device, dtype=dtype),
            "clip_eps": clip_eps,
            "beta": beta,
            "B": B,
            "S": S,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._make_test_case(
            1,
            4,
            advantages=[[1.0, -2.0, 3.0, -4.0]],
            log_pi=[[0.262364, -0.356675, 0.0953102, -0.223144]],
            log_pi_old=[[0.0, 0.0, 0.0, 0.0]],
            log_ref=[[0.1, -0.2, 0.05, -0.1]],
            clip_eps=0.2,
            beta=0.01,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Hand-computed clipping example with positive and negative advantages.
        tests.append(self.generate_example_test())

        # Single element with no policy or KL change.
        tests.append(
            self._make_test_case(
                1,
                1,
                advantages=[[2.5]],
                log_pi=[[0.0]],
                log_pi_old=[[0.0]],
                log_ref=[[0.0]],
            )
        )

        # Zero advantages isolate the non-negative reference-policy KL regularizer.
        tests.append(
            self._make_test_case(
                1,
                1,
                advantages=[[0.0]],
                log_pi=[[-1.0]],
                log_pi_old=[[-1.0]],
                log_ref=[[-2.0]],
                beta=1.0,
            )
        )

        # Zero objective inputs.
        tests.append(
            self._make_test_case(
                2,
                4,
                advantages=[[0.0] * 4, [0.0] * 4],
                log_pi=[[0.0] * 4, [0.0] * 4],
                log_pi_old=[[0.0] * 4, [0.0] * 4],
                log_ref=[[0.0] * 4, [0.0] * 4],
            )
        )

        # Ratios exactly at both clipping boundaries.
        tests.append(
            self._make_test_case(
                1,
                4,
                advantages=[[1.0, -1.0, 2.0, -2.0]],
                log_pi=[[math.log(1.2), math.log(0.8), 0.0, 0.0]],
                log_pi_old=[[0.0] * 4],
                log_ref=[[0.0] * 4],
            )
        )

        # Power-of-two shape with random mixed-sign values.
        tests.append(self._make_test_case(4, 16))
        tests.append(self._make_test_case(8, 64, clip_eps=0.1, beta=0.05))

        # Non-power-of-two shapes.
        tests.append(self._make_test_case(3, 30))
        tests.append(self._make_test_case(5, 100, clip_eps=0.3))

        # Realistic PPO rollout shape.
        tests.append(self._make_test_case(16, 512))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        B, S = 256, 16384
        return {
            "advantages": RandnTensor((B, S)),
            "log_pi": RandnTensor((B, S)),
            "log_pi_old": RandnTensor((B, S)),
            "log_ref": RandnTensor((B, S)),
            "output": OutTensor((1,)),
            "clip_eps": 0.2,
            "beta": 0.01,
            "B": B,
            "S": S,
        }
