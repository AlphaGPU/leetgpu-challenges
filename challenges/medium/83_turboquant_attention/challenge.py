import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="TurboQuant KV Cache Attention",
            atol=1e-3,
            rtol=1e-3,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        Q: torch.Tensor,
        K_idx: torch.Tensor,
        Pi: torch.Tensor,
        codebook: torch.Tensor,
        scores: torch.Tensor,
        B: int,
        S: int,
        D: int,
        C: int,
    ):
        assert Q.shape == (B, D)
        assert K_idx.shape == (S, D)
        assert Pi.shape == (D, D)
        assert codebook.shape == (C,)
        assert scores.shape == (B, S)
        assert Q.dtype == torch.float32
        assert K_idx.dtype == torch.uint8
        assert Pi.dtype == torch.float32
        assert codebook.dtype == torch.float32
        assert scores.dtype == torch.float32
        assert Q.device.type == "cuda"
        assert K_idx.device.type == "cuda"
        assert Pi.device.type == "cuda"
        assert codebook.device.type == "cuda"
        assert scores.device.type == "cuda"

        # Dequantize keys: lookup centroids then rotate back
        K_centroids = codebook[K_idx.long()]  # S x D
        K_deq = K_centroids @ Pi  # S x D

        # Compute attention scores
        scores.copy_(Q @ K_deq.T)  # B x S

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_idx": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "Pi": (ctypes.POINTER(ctypes.c_float), "in"),
            "codebook": (ctypes.POINTER(ctypes.c_float), "in"),
            "scores": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
        }

    def _make_rotation(self, D):
        G = torch.randn(D, D, device="cuda")
        Q, _ = torch.linalg.qr(G)
        return Q

    def _make_codebook(self, C, scale=1.0):
        return torch.linspace(-scale, scale, C, device="cuda", dtype=torch.float32)

    def generate_example_test(self) -> Dict[str, Any]:
        B, S, D, C = 2, 3, 2, 4
        Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda", dtype=torch.float32)
        K_idx = torch.tensor([[0, 3], [1, 2], [3, 0]], device="cuda", dtype=torch.uint8)
        Pi = torch.eye(D, device="cuda", dtype=torch.float32)
        codebook = torch.tensor([-0.75, -0.25, 0.25, 0.75], device="cuda", dtype=torch.float32)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        return {
            "Q": Q,
            "K_idx": K_idx,
            "Pi": Pi,
            "codebook": codebook,
            "scores": scores,
            "B": B,
            "S": S,
            "D": D,
            "C": C,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge case: single query, single key, D=1, C=2
        B, S, D, C = 1, 1, 1, 2
        Q = torch.tensor([[0.5]], device="cuda", dtype=torch.float32)
        K_idx = torch.tensor([[1]], device="cuda", dtype=torch.uint8)
        Pi = torch.eye(D, device="cuda", dtype=torch.float32)
        codebook = self._make_codebook(C)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "Q": Q,
                "K_idx": K_idx,
                "Pi": Pi,
                "codebook": codebook,
                "scores": scores,
                "B": B,
                "S": S,
                "D": D,
                "C": C,
            }
        )

        # Edge case: zeros query
        B, S, D, C = 2, 3, 4, 4
        Q = torch.zeros(B, D, device="cuda", dtype=torch.float32)
        K_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        Pi = self._make_rotation(D)
        codebook = self._make_codebook(C)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "Q": Q,
                "K_idx": K_idx,
                "Pi": Pi,
                "codebook": codebook,
                "scores": scores,
                "B": B,
                "S": S,
                "D": D,
                "C": C,
            }
        )

        # Edge case: negative query values
        B, S, D, C = 2, 4, 4, 4
        Q = torch.tensor(
            [[-0.5, -0.3, -0.8, -0.1], [-1.0, -0.5, -0.2, -0.9]],
            device="cuda",
            dtype=torch.float32,
        )
        K_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        Pi = self._make_rotation(D)
        codebook = self._make_codebook(C)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "Q": Q,
                "K_idx": K_idx,
                "Pi": Pi,
                "codebook": codebook,
                "scores": scores,
                "B": B,
                "S": S,
                "D": D,
                "C": C,
            }
        )

        # Power-of-2: B=4, S=16, D=32, C=8
        B, S, D, C = 4, 16, 32, 8
        Q = torch.randn(B, D, device="cuda", dtype=torch.float32) * 0.5
        K_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        Pi = self._make_rotation(D)
        codebook = self._make_codebook(C, scale=1.5)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "Q": Q,
                "K_idx": K_idx,
                "Pi": Pi,
                "codebook": codebook,
                "scores": scores,
                "B": B,
                "S": S,
                "D": D,
                "C": C,
            }
        )

        # Power-of-2: B=8, S=64, D=64, C=16
        B, S, D, C = 8, 64, 64, 16
        Q = torch.randn(B, D, device="cuda", dtype=torch.float32) * 0.3
        K_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        Pi = self._make_rotation(D)
        codebook = self._make_codebook(C)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "Q": Q,
                "K_idx": K_idx,
                "Pi": Pi,
                "codebook": codebook,
                "scores": scores,
                "B": B,
                "S": S,
                "D": D,
                "C": C,
            }
        )

        # Power-of-2: B=16, S=128, D=128, C=16
        B, S, D, C = 16, 128, 128, 16
        Q = torch.randn(B, D, device="cuda", dtype=torch.float32) * 0.3
        K_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        Pi = self._make_rotation(D)
        codebook = self._make_codebook(C)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "Q": Q,
                "K_idx": K_idx,
                "Pi": Pi,
                "codebook": codebook,
                "scores": scores,
                "B": B,
                "S": S,
                "D": D,
                "C": C,
            }
        )

        # Non-power-of-2: B=3, S=30, D=50, C=8
        B, S, D, C = 3, 30, 50, 8
        Q = torch.randn(B, D, device="cuda", dtype=torch.float32) * 0.4
        K_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        Pi = self._make_rotation(D)
        codebook = self._make_codebook(C)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "Q": Q,
                "K_idx": K_idx,
                "Pi": Pi,
                "codebook": codebook,
                "scores": scores,
                "B": B,
                "S": S,
                "D": D,
                "C": C,
            }
        )

        # Non-power-of-2: B=7, S=255, D=100, C=16
        B, S, D, C = 7, 255, 100, 16
        Q = torch.randn(B, D, device="cuda", dtype=torch.float32) * 0.6
        K_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        Pi = self._make_rotation(D)
        codebook = self._make_codebook(C, scale=1.5)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "Q": Q,
                "K_idx": K_idx,
                "Pi": Pi,
                "codebook": codebook,
                "scores": scores,
                "B": B,
                "S": S,
                "D": D,
                "C": C,
            }
        )

        # Realistic: B=16, S=4096, D=128, C=16
        B, S, D, C = 16, 4096, 128, 16
        Q = torch.randn(B, D, device="cuda", dtype=torch.float32) * 0.3
        K_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        Pi = self._make_rotation(D)
        codebook = self._make_codebook(C)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "Q": Q,
                "K_idx": K_idx,
                "Pi": Pi,
                "codebook": codebook,
                "scores": scores,
                "B": B,
                "S": S,
                "D": D,
                "C": C,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        B, S, D, C = 32, 32768, 128, 16
        Q = torch.randn(B, D, device="cuda", dtype=torch.float32) * 0.3
        K_idx = torch.randint(0, C, (S, D), device="cuda", dtype=torch.uint8)
        Pi = self._make_rotation(D)
        codebook = self._make_codebook(C)
        scores = torch.zeros(B, S, device="cuda", dtype=torch.float32)
        return {
            "Q": Q,
            "K_idx": K_idx,
            "Pi": Pi,
            "codebook": codebook,
            "scores": scores,
            "B": B,
            "S": S,
            "D": D,
            "C": C,
        }
