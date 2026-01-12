import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Weight Dequantization", atol=1e-05, rtol=1e-05, num_gpus=1, access_tier="free"
        )

    def reference_impl(self, X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, BLOCK_SIZE: int):
        # S shape: (ceil(M/BLOCK_SIZE), ceil(N/BLOCK_SIZE))
        # We expand S to match X's shape (M, N)

        # Expand rows
        S_expanded = S.repeat_interleave(BLOCK_SIZE, dim=0)
        # Crop if M is not a multiple of BLOCK_SIZE
        if S_expanded.shape[0] > M:
            S_expanded = S_expanded[:M, :]

        # Expand cols
        S_expanded = S_expanded.repeat_interleave(BLOCK_SIZE, dim=1)
        # Crop if N is not a multiple of BLOCK_SIZE
        if S_expanded.shape[1] > N:
            S_expanded = S_expanded[:, :N]

        # Perform element-wise multiplication
        # Ensure Y is updated in-place
        Y.copy_(X.to(Y.dtype) * S_expanded.to(Y.dtype))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "S": (ctypes.POINTER(ctypes.c_float), "in"),
            "Y": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "BLOCK_SIZE": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        M, N = 256, 256
        BLOCK_SIZE = 128
        X = torch.randn(M, N, device="cuda", dtype=torch.float32)
        # S shape
        s_rows = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        s_cols = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        S = torch.randn(s_rows, s_cols, device="cuda", dtype=torch.float32)
        Y = torch.empty_like(X)

        return {
            "X": X,
            "S": S,
            "Y": Y,
            "M": M,
            "N": N,
            "BLOCK_SIZE": BLOCK_SIZE,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Case 1: Perfect Multiple
        M, N = 256, 256
        BLOCK_SIZE = 128
        tests.append(
            {
                "name": "perfect_multiple",
                "X": torch.randn(M, N, device="cuda", dtype=torch.float32),
                "S": torch.randn(2, 2, device="cuda", dtype=torch.float32),
                "Y": torch.zeros(M, N, device="cuda", dtype=torch.float32),
                "M": M,
                "N": N,
                "BLOCK_SIZE": BLOCK_SIZE,
            }
        )

        # Case 2: Odd sizes (padding needed)
        M, N = 130, 200
        BLOCK_SIZE = 128
        # Rows: ceil(130/128) = 2. Cols: ceil(200/128) = 2.
        tests.append(
            {
                "name": "irregular_size",
                "X": torch.randn(M, N, device="cuda", dtype=torch.float32),
                "S": torch.randn(2, 2, device="cuda", dtype=torch.float32),
                "Y": torch.zeros(M, N, device="cuda", dtype=torch.float32),
                "M": M,
                "N": N,
                "BLOCK_SIZE": BLOCK_SIZE,
            }
        )

        # Case 3: Small Block Size
        M, N = 64, 64
        BLOCK_SIZE = 32
        tests.append(
            {
                "name": "small_blocks",
                "X": torch.randn(M, N, device="cuda", dtype=torch.float32),
                "S": torch.randn(2, 2, device="cuda", dtype=torch.float32),
                "Y": torch.zeros(M, N, device="cuda", dtype=torch.float32),
                "M": M,
                "N": N,
                "BLOCK_SIZE": BLOCK_SIZE,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M, N = 8192, 8192
        BLOCK_SIZE = 128
        X = torch.randn(M, N, device="cuda", dtype=torch.float32)
        s_rows = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        s_cols = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        S = torch.randn(s_rows, s_cols, device="cuda", dtype=torch.float32)
        Y = torch.empty_like(X)

        return {
            "X": X,
            "S": S,
            "Y": Y,
            "M": M,
            "N": N,
            "BLOCK_SIZE": BLOCK_SIZE,
        }
