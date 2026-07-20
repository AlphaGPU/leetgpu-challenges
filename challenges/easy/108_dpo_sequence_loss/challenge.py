import ctypes
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase, OutTensor, RandnTensor


class Challenge(ChallengeBase):
    name = "DPO Sequence Loss"
    atol = 0.001
    rtol = 0.001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        chosen_ref_logps: torch.Tensor,
        rejected_ref_logps: torch.Tensor,
        output: torch.Tensor,
        beta: float,
        B: int,
    ):
        assert chosen_logps.shape == (B,)
        assert rejected_logps.shape == (B,)
        assert chosen_ref_logps.shape == (B,)
        assert rejected_ref_logps.shape == (B,)
        assert output.shape == (1,)
        assert (
            chosen_logps.dtype
            == rejected_logps.dtype
            == chosen_ref_logps.dtype
            == rejected_ref_logps.dtype
            == output.dtype
            == torch.float32
        )

        chosen_margin = chosen_logps - rejected_logps
        reference_margin = chosen_ref_logps - rejected_ref_logps
        logits = beta * (chosen_margin - reference_margin)
        output[0] = -F.logsigmoid(logits).mean()

    def reference_impl_jax(
        self, chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, beta, B,
    ):
        import jax
        import jax.numpy as jnp

        chosen_margin = chosen_logps - rejected_logps
        reference_margin = chosen_ref_logps - rejected_ref_logps
        logits = beta * (chosen_margin - reference_margin)
        return jnp.mean(jax.nn.softplus(-logits))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "chosen_logps": (ctypes.POINTER(ctypes.c_float), "in"),
            "rejected_logps": (ctypes.POINTER(ctypes.c_float), "in"),
            "chosen_ref_logps": (ctypes.POINTER(ctypes.c_float), "in"),
            "rejected_ref_logps": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "beta": (ctypes.c_float, "in"),
            "B": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        B,
        chosen_logps=None,
        rejected_logps=None,
        chosen_ref_logps=None,
        rejected_ref_logps=None,
        beta=0.1,
    ):
        dtype = torch.float32
        device = self.device

        def tensor_or_random(values):
            if values is None:
                return torch.randn(B, device=device, dtype=dtype)
            return torch.tensor(values, device=device, dtype=dtype)

        return {
            "chosen_logps": tensor_or_random(chosen_logps),
            "rejected_logps": tensor_or_random(rejected_logps),
            "chosen_ref_logps": tensor_or_random(chosen_ref_logps),
            "rejected_ref_logps": tensor_or_random(rejected_ref_logps),
            "output": torch.empty(1, device=device, dtype=dtype),
            "beta": beta,
            "B": B,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._make_test_case(
            4,
            chosen_logps=[0.0, 1.0, -1.0, 2.0],
            rejected_logps=[0.0, 0.0, 0.0, 0.0],
            chosen_ref_logps=[0.0, 0.0, 0.0, 0.0],
            rejected_ref_logps=[0.0, 0.0, 0.0, 0.0],
            beta=0.1,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Hand-computed example with positive, negative, and zero logits.
        tests.append(self.generate_example_test())

        # Single sequence with a zero preference margin.
        tests.append(
            self._make_test_case(
                1,
                chosen_logps=[-2.0],
                rejected_logps=[-2.0],
                chosen_ref_logps=[-1.0],
                rejected_ref_logps=[-1.0],
            )
        )

        # Non-zero reference margins with positive and negative DPO logits.
        tests.append(
            self._make_test_case(
                2,
                chosen_logps=[2.0, -1.0],
                rejected_logps=[0.0, 1.0],
                chosen_ref_logps=[0.5, -0.5],
                rejected_ref_logps=[-0.5, 0.5],
                beta=0.7,
            )
        )

        # All zero inputs: loss is log(2).
        tests.append(
            self._make_test_case(
                8,
                beta=0.1,
                chosen_logps=[0.0] * 8,
                rejected_logps=[0.0] * 8,
                chosen_ref_logps=[0.0] * 8,
                rejected_ref_logps=[0.0] * 8,
            )
        )

        # Large positive and negative logits expose sigmoid/log underflow.
        tests.append(
            self._make_test_case(
                4,
                chosen_logps=[1000.0, -1000.0, 500.0, -500.0],
                rejected_logps=[0.0] * 4,
                chosen_ref_logps=[0.0] * 4,
                rejected_ref_logps=[0.0] * 4,
            )
        )

        # Beta=0 removes the preference margin and produces log(2).
        tests.append(self._make_test_case(16, beta=0.0))

        # Power-of-two and non-power-of-two batch sizes.
        tests.append(self._make_test_case(64))
        tests.append(self._make_test_case(127, beta=0.05))

        # Realistic accumulated batch.
        tests.append(self._make_test_case(4096, beta=0.2))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        B = 65536
        return {
            "chosen_logps": RandnTensor((B,)),
            "rejected_logps": RandnTensor((B,)),
            "chosen_ref_logps": RandnTensor((B,)),
            "rejected_ref_logps": RandnTensor((B,)),
            "output": OutTensor((1,)),
            "beta": 0.1,
            "B": B,
        }
