import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused AdamW"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        params: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        N: int,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        t: int,
    ):
        assert params.shape == (N,)
        assert grad.shape == (N,)
        assert m.shape == (N,)
        assert v.shape == (N,)
        assert params.dtype == grad.dtype == m.dtype == v.dtype == torch.float32
        assert t >= 1

        # Decoupled weight decay: applied directly to the parameters, not folded
        # into the gradient (this is what distinguishes AdamW from Adam + L2 reg).
        params.mul_(1.0 - lr * weight_decay)

        # Update biased first and second raw moment estimates.
        m.copy_(beta1 * m + (1.0 - beta1) * grad)
        v.copy_(beta2 * v + (1.0 - beta2) * grad * grad)

        # Bias-correct the moment estimates.
        bias_correction1 = 1.0 - beta1**t
        bias_correction2 = 1.0 - beta2**t
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2

        # Adaptive step.
        params.sub_(lr * m_hat / (v_hat.sqrt() + eps))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "params": (ctypes.POINTER(ctypes.c_float), "inout"),
            "grad": (ctypes.POINTER(ctypes.c_float), "in"),
            "m": (ctypes.POINTER(ctypes.c_float), "inout"),
            "v": (ctypes.POINTER(ctypes.c_float), "inout"),
            "N": (ctypes.c_int, "in"),
            "lr": (ctypes.c_float, "in"),
            "beta1": (ctypes.c_float, "in"),
            "beta2": (ctypes.c_float, "in"),
            "eps": (ctypes.c_float, "in"),
            "weight_decay": (ctypes.c_float, "in"),
            "t": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        N,
        lr=0.1,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        t=1,
        zero_state=True,
        zero_grad=False,
        param_range=(-2.0, 2.0),
        grad_range=(-1.0, 1.0),
        seed=0,
    ):
        device = self.device
        dtype = torch.float32
        gen = torch.Generator(device=device).manual_seed(seed)

        params = torch.empty(N, device=device, dtype=dtype).uniform_(*param_range, generator=gen)
        if zero_grad:
            grad = torch.zeros(N, device=device, dtype=dtype)
        else:
            grad = torch.empty(N, device=device, dtype=dtype).uniform_(*grad_range, generator=gen)
        if zero_state:
            m = torch.zeros(N, device=device, dtype=dtype)
            v = torch.zeros(N, device=device, dtype=dtype)
        else:
            # Simulate an optimizer that has already taken some steps: m carries
            # the sign of typical gradients, v is a small positive magnitude.
            m = torch.empty(N, device=device, dtype=dtype).uniform_(-0.5, 0.5, generator=gen)
            v = torch.empty(N, device=device, dtype=dtype).uniform_(0.0, 0.5, generator=gen)

        return {
            "params": params,
            "grad": grad,
            "m": m,
            "v": v,
            "N": N,
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
            "t": t,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        params = torch.tensor([1.0, -1.0], device=device, dtype=dtype)
        grad = torch.tensor([0.1, -0.1], device=device, dtype=dtype)
        m = torch.zeros(2, device=device, dtype=dtype)
        v = torch.zeros(2, device=device, dtype=dtype)
        return {
            "params": params,
            "grad": grad,
            "m": m,
            "v": v,
            "N": 2,
            "lr": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "t": 1,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Single element, first step, fresh (zero) optimizer state.
        tests.append(self._make_test_case(N=1, t=1, zero_state=True, seed=1))

        # Zero gradient, fresh (zero) optimizer state: m and v stay exactly zero,
        # so the adaptive step is zero and only weight decay moves the parameters.
        tests.append(self._make_test_case(N=8, t=5, zero_state=True, zero_grad=True, seed=2))

        # Zero gradient with warm (nonzero) prior state: no new gradient signal
        # arrives, but existing momentum still drives an adaptive step while m
        # and v exponentially decay toward zero.
        tests.append(self._make_test_case(N=8, t=5, zero_state=False, zero_grad=True, seed=11))

        # Zero weight decay: reduces to plain Adam.
        tests.append(self._make_test_case(N=16, t=1, weight_decay=0.0, zero_state=True, seed=3))

        # Warm state, mid-training step count: bias correction is neither ~0 nor ~1.
        tests.append(self._make_test_case(N=32, t=50, zero_state=False, seed=4))

        # Very large step count: bias corrections both saturate close to 1.
        tests.append(self._make_test_case(N=16, t=100000, zero_state=False, seed=5))

        # Negative parameters and negative gradients throughout.
        tests.append(
            self._make_test_case(
                N=16,
                t=1,
                zero_state=True,
                param_range=(-2.0, -0.5),
                grad_range=(-1.0, -0.2),
                seed=6,
            )
        )

        # Non-default betas: catches implementations that hardcode 0.9 / 0.999.
        tests.append(
            self._make_test_case(N=64, t=10, beta1=0.8, beta2=0.99, zero_state=False, seed=7)
        )

        # Larger eps: shifts the denominator enough to matter numerically.
        tests.append(self._make_test_case(N=64, t=1, eps=1e-2, zero_state=True, seed=8))

        # Very small learning rate: update should be tiny but nonzero.
        tests.append(self._make_test_case(N=64, t=1, lr=1e-6, zero_state=True, seed=9))

        # Larger, realistic-ish size with default hyperparameters.
        tests.append(self._make_test_case(N=1024, t=200, zero_state=False, seed=10))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # 4096 x 4096 weight matrix flattened (LLaMA-7B-style hidden size), a
        # realistic single-tensor slice of an optimizer step over a large model.
        return self._make_test_case(N=4096 * 4096, t=1000, zero_state=False, seed=0)
