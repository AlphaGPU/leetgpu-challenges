import ctypes
import math
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
        qjl_signs: torch.Tensor,
        gamma: torch.Tensor,
        Pi: torch.Tensor,
        S_mat: torch.Tensor,
        codebook: torch.Tensor,
        scores: torch.Tensor,
        B: int,
        S: int,
        D: int,
        C: int,
    ):
        assert Q.shape == (B, D)
        assert K_idx.shape == (S, D)
        assert qjl_signs.shape == (S, D)
        assert gamma.shape == (S,)
        assert Pi.shape == (D, D)
        assert S_mat.shape == (D, D)
        assert codebook.shape == (C,)
        assert scores.shape == (B, S)
        assert Q.dtype == torch.float32
        assert K_idx.dtype == torch.uint8
        assert qjl_signs.dtype == torch.int8
        assert gamma.dtype == torch.float32
        assert Pi.dtype == torch.float32
        assert S_mat.dtype == torch.float32
        assert codebook.dtype == torch.float32
        assert scores.dtype == torch.float32
        assert Q.device.type == "cuda"
        assert K_idx.device.type == "cuda"
        assert qjl_signs.device.type == "cuda"
        assert gamma.device.type == "cuda"
        assert Pi.device.type == "cuda"
        assert S_mat.device.type == "cuda"
        assert codebook.device.type == "cuda"
        assert scores.device.type == "cuda"

        # Stage 1: MSE dequantization — lookup centroids, rotate back
        K_centroids = codebook[K_idx.long()]  # [S, D]
        K_mse = K_centroids @ Pi  # [S, D]  (row convention: ỹ @ Π = Π^T · ỹ)

        # Stage 2: QJL dequantization — reconstruct residual correction
        scale = math.sqrt(math.pi / 2.0) / D
        K_qjl = scale * gamma.unsqueeze(1) * (qjl_signs.float() @ S_mat)  # [S, D]

        # Combined dequantization
        K_deq = K_mse + K_qjl  # [S, D]

        # Attention scores
        scores.copy_(Q @ K_deq.T)  # [B, S]

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_idx": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "qjl_signs": (ctypes.POINTER(ctypes.c_int8), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "Pi": (ctypes.POINTER(ctypes.c_float), "in"),
            "S_mat": (ctypes.POINTER(ctypes.c_float), "in"),
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

    def _encode_keys(self, K, Pi, S_mat, codebook):
        """Simulate TurboQuant_prod encoding: rotate, quantize, compute QJL on residual."""
        S, D = K.shape

        # Stage 1: MSE encoding
        Y = K @ Pi.T  # rotate into quantization space
        # Scalar quantize each coordinate to nearest centroid
        diffs = Y.unsqueeze(-1) - codebook.unsqueeze(0).unsqueeze(0)  # [S, D, C]
        K_idx = diffs.abs().argmin(dim=-1).to(torch.uint8)  # [S, D]

        # MSE dequantization (to compute residual)
        K_centroids = codebook[K_idx.long()]  # [S, D]
        K_mse = K_centroids @ Pi  # [S, D]

        # Stage 2: QJL encoding of residual
        residual = K - K_mse  # [S, D]
        gamma = residual.norm(dim=1)  # [S]
        proj = residual @ S_mat.T  # [S, D] (row convention for S · r)
        qjl_signs = torch.sign(proj).to(torch.int8)  # [S, D]
        # Ensure no zeros (sign(0)=0 → map to +1)
        qjl_signs[qjl_signs == 0] = 1

        return K_idx, qjl_signs, gamma

    def _make_test_case(self, B, S_seq, D, C, zero_q=False, seed=42):
        torch.manual_seed(seed)
        device = "cuda"

        Pi = self._make_rotation(D)
        S_mat = torch.randn(D, D, device=device, dtype=torch.float32)
        codebook = self._make_codebook(C)

        if zero_q:
            Q = torch.zeros(B, D, device=device, dtype=torch.float32)
        else:
            Q = torch.randn(B, D, device=device, dtype=torch.float32) * 0.5

        # Generate realistic keys and encode them
        K = torch.randn(S_seq, D, device=device, dtype=torch.float32) * 0.3
        K_idx, qjl_signs, gamma = self._encode_keys(K, Pi, S_mat, codebook)

        scores = torch.zeros(B, S_seq, device=device, dtype=torch.float32)

        return {
            "Q": Q,
            "K_idx": K_idx,
            "qjl_signs": qjl_signs,
            "gamma": gamma,
            "Pi": Pi,
            "S_mat": S_mat,
            "codebook": codebook,
            "scores": scores,
            "B": B,
            "S": S_seq,
            "D": D,
            "C": C,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        B, S, D, C = 2, 3, 2, 4

        Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=torch.float32)
        K_idx = torch.tensor([[0, 3], [1, 2], [3, 0]], device=device, dtype=torch.uint8)
        # QJL signs: all +1 for simplicity
        qjl_signs = torch.ones(S, D, device=device, dtype=torch.int8)
        # gamma = 0: no QJL correction (reduces to MSE-only for this example)
        gamma = torch.zeros(S, device=device, dtype=torch.float32)
        Pi = torch.eye(D, device=device, dtype=torch.float32)
        S_mat = torch.eye(D, device=device, dtype=torch.float32)
        codebook = torch.tensor([-0.75, -0.25, 0.25, 0.75], device=device, dtype=torch.float32)
        scores = torch.zeros(B, S, device=device, dtype=torch.float32)

        return {
            "Q": Q,
            "K_idx": K_idx,
            "qjl_signs": qjl_signs,
            "gamma": gamma,
            "Pi": Pi,
            "S_mat": S_mat,
            "codebook": codebook,
            "scores": scores,
            "B": B,
            "S": S,
            "D": D,
            "C": C,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge: single query, single key, D=1
        tests.append(self._make_test_case(1, 1, 1, 2, seed=1))

        # Edge: zero query
        tests.append(self._make_test_case(2, 3, 4, 4, zero_q=True, seed=2))

        # Edge: small with negative queries
        tests.append(self._make_test_case(2, 4, 4, 4, seed=3))

        # Power-of-2: B=4, S=16, D=32, C=8
        tests.append(self._make_test_case(4, 16, 32, 8, seed=10))

        # Power-of-2: B=8, S=64, D=64, C=16
        tests.append(self._make_test_case(8, 64, 64, 16, seed=20))

        # Power-of-2: B=16, S=128, D=128, C=16
        tests.append(self._make_test_case(16, 128, 128, 16, seed=30))

        # Non-power-of-2: B=3, S=30, D=50, C=8
        tests.append(self._make_test_case(3, 30, 50, 8, seed=40))

        # Non-power-of-2: B=7, S=255, D=100, C=16
        tests.append(self._make_test_case(7, 255, 100, 16, seed=50))

        # Realistic: B=16, S=4096, D=128, C=16
        tests.append(self._make_test_case(16, 4096, 128, 16, seed=60))

        # Realistic: B=32, S=8192, D=128, C=8
        tests.append(self._make_test_case(32, 8192, 128, 8, seed=70))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test_case(32, 32768, 128, 16, seed=0)
