import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused MoE SwiGLU Layer"
    # Outputs are O(1) sums of three chained float32 GEMMs (K up to 1,024) plus a
    # weighted top-k combine. Accumulation-order error stays around 1e-5; routing a
    # token to the wrong expert, dropping a weight or the second expert changes
    # outputs by O(0.1), so 1e-3 is safe in both directions.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        output: torch.Tensor,
        T: int,
        H: int,
        I: int,  # noqa: E741
        E: int,
        K: int,
    ):
        assert x.shape == (T, H)
        assert topk_ids.shape == (T, K)
        assert topk_weights.shape == (T, K)
        assert w_gate.shape == (E, H, I)
        assert w_up.shape == (E, H, I)
        assert w_down.shape == (E, I, H)
        assert output.shape == (T, H)
        assert topk_ids.dtype == torch.int32
        assert x.dtype == topk_weights.dtype == w_gate.dtype == torch.float32
        assert w_up.dtype == w_down.dtype == output.dtype == torch.float32

        ids = topk_ids.long()
        result = torch.zeros(T, H, device=x.device, dtype=torch.float32)
        for e in range(E):
            # Per-token combine weight for expert e (0 if the token is not routed to e).
            coef = (topk_weights * (ids == e).to(torch.float32)).sum(dim=1, keepdim=True)
            h = torch.nn.functional.silu(x @ w_gate[e]) * (x @ w_up[e])
            result = result + coef * (h @ w_down[e])
        output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "topk_ids": (ctypes.POINTER(ctypes.c_int), "in"),
            "topk_weights": (ctypes.POINTER(ctypes.c_float), "in"),
            "w_gate": (ctypes.POINTER(ctypes.c_float), "in"),
            "w_up": (ctypes.POINTER(ctypes.c_float), "in"),
            "w_down": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "T": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "I": (ctypes.c_int, "in"),
            "E": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def _make(self, T, H, I, E, K, seed=0, ids=None, weights=None):  # noqa: E741
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        x = torch.empty(T, H, device=dev, dtype=f32).uniform_(-1.0, 1.0)
        w_gate = torch.empty(E, H, I, device=dev, dtype=f32).uniform_(-1.0, 1.0) / (H**0.5)
        w_up = torch.empty(E, H, I, device=dev, dtype=f32).uniform_(-1.0, 1.0) / (H**0.5)
        w_down = torch.empty(E, I, H, device=dev, dtype=f32).uniform_(-1.0, 1.0) / (I**0.5)
        if ids is None:
            # K distinct experts per token: argsort of random scores.
            topk_ids = torch.argsort(torch.rand(T, E, device=dev), dim=1)[:, :K].to(torch.int32)
        else:
            topk_ids = torch.tensor(ids, device=dev, dtype=torch.int32)
        if weights is None:
            topk_weights = torch.softmax(torch.randn(T, K, device=dev, dtype=f32), dim=1)
        else:
            topk_weights = torch.tensor(weights, device=dev, dtype=f32)
        return {
            "x": x,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "w_gate": w_gate,
            "w_up": w_up,
            "w_down": w_down,
            "output": torch.zeros(T, H, device=dev, dtype=f32),
            "T": T,
            "H": H,
            "I": I,
            "E": E,
            "K": K,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dev, f32 = self.device, torch.float32
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=dev, dtype=f32)
        topk_ids = torch.tensor([[0], [1]], device=dev, dtype=torch.int32)
        topk_weights = torch.tensor([[1.0], [0.5]], device=dev, dtype=f32)
        eye = torch.eye(2, device=dev, dtype=f32)
        w_gate = torch.stack([eye, eye])
        w_up = torch.stack([eye, 2.0 * eye])
        w_down = torch.stack([eye, eye])
        return {
            "x": x,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "w_gate": w_gate,
            "w_up": w_up,
            "w_down": w_down,
            "output": torch.zeros(2, 2, device=dev, dtype=f32),
            "T": 2,
            "H": 2,
            "I": 2,
            "E": 2,
            "K": 1,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]

        # Single token, single expert, top-1.
        tests.append(self._make(1, 4, 4, 1, 1, seed=1))

        # Zero input: SiLU(0) * 0 = 0, output must be all zeros.
        z = self._make(3, 8, 16, 4, 2, seed=2)
        z["x"].zero_()
        tests.append(z)

        # Every token routed to the same pair of experts; experts 2-3 receive no tokens
        # (catches kernels that assume every expert has work or mis-size empty groups).
        tests.append(
            self._make(
                4, 16, 32, 4, 2, seed=3, ids=[[1, 0]] * 4, weights=[[0.7, 0.3], [0.5, 0.5]] * 2
            )
        )

        # Unnormalized and negative combine weights: kernels must use the given weights
        # as-is rather than renormalizing or taking a softmax.
        tests.append(
            self._make(
                3,
                8,
                8,
                4,
                2,
                seed=4,
                ids=[[3, 2], [0, 3], [2, 1]],
                weights=[[2.0, -1.0], [0.0, 1.5], [-0.25, 0.25]],
            )
        )

        # Negative-only inputs.
        neg = self._make(16, 32, 64, 8, 2, seed=5)
        neg["x"].abs_().neg_()
        tests.append(neg)

        # Non-power-of-two dims (tile tails in T, H and I).
        tests.append(self._make(30, 100, 70, 6, 3, seed=6))

        # Power-of-two dims, top-2 of 8.
        tests.append(self._make(128, 128, 256, 8, 2, seed=7))

        # Top-K equal to E (dense mixture).
        tests.append(self._make(64, 64, 64, 4, 4, seed=8))

        # Realistic size.
        tests.append(self._make(1024, 512, 1024, 8, 2, seed=9))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make(4096, 1024, 1408, 16, 2, seed=0)
