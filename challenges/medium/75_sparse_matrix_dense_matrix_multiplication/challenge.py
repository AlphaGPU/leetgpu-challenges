import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Sparse Matrix-Dense Matrix Multiplication",
            atol=1e-03,
            rtol=1e-03,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        M: int,
        N: int,
        K: int,
        nnz: int,
    ):
        if A.shape == (M * N,):
            A_matrix = A.view(M, N)
        elif A.shape == (M, N):
            A_matrix = A
        else:
            raise AssertionError(
                f"A.shape {A.shape} does not match expected {(M * N,)} or {(M, N)}"
            )
        if B.shape == (N * K,):
            B_matrix = B.view(N, K)
        elif B.shape == (N, K):
            B_matrix = B
        else:
            raise AssertionError(
                f"B.shape {B.shape} does not match expected {(N * K,)} or {(N, K)}"
            )
        assert C.shape == (M, K) or C.shape == (
            M * K,
        ), f"C.shape {C.shape} does not match expected {(M, K)} or {(M * K,)}"
        assert A_matrix.dtype == torch.float32
        assert B_matrix.dtype == torch.float32
        assert A_matrix.device.type == "cuda"
        assert B_matrix.device.type == "cuda"
        assert C.device.type == "cuda"
        result = torch.matmul(A_matrix, B_matrix)
        C.copy_(result.view(C.shape))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "nnz": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor(
            [
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
            ],
            device="cuda",
            dtype=dtype,
        )
        B = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            device="cuda",
            dtype=dtype,
        )
        C = torch.empty((3, 2), device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "M": 3,
            "N": 4,
            "K": 2,
            "nnz": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # edge_1x1x1
        tests.append(
            {
                "A": torch.tensor([[3.0]], device="cuda", dtype=dtype),
                "B": torch.tensor([[2.0]], device="cuda", dtype=dtype),
                "C": torch.empty((1, 1), device="cuda", dtype=dtype),
                "M": 1,
                "N": 1,
                "K": 1,
                "nnz": 1,
            }
        )

        # edge_2x2_k1_spmv_like
        tests.append(
            {
                "A": torch.tensor([[1.0, 0.0], [0.0, 2.0]], device="cuda", dtype=dtype),
                "B": torch.tensor([[3.0], [4.0]], device="cuda", dtype=dtype),
                "C": torch.empty((2, 1), device="cuda", dtype=dtype),
                "M": 2,
                "N": 2,
                "K": 1,
                "nnz": 2,
            }
        )

        # edge_zero_matrix
        tests.append(
            {
                "A": torch.zeros((3, 3), device="cuda", dtype=dtype),
                "B": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda", dtype=dtype),
                "C": torch.empty((3, 2), device="cuda", dtype=dtype),
                "M": 3,
                "N": 3,
                "K": 2,
                "nnz": 0,
            }
        )

        # edge_identity_a
        tests.append(
            {
                "A": torch.eye(4, device="cuda", dtype=dtype),
                "B": torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    device="cuda",
                    dtype=dtype,
                ),
                "C": torch.empty((4, 3), device="cuda", dtype=dtype),
                "M": 4,
                "N": 4,
                "K": 3,
                "nnz": 4,
            }
        )

        # power_of_2_16x16x8
        M, N, K = 16, 16, 8
        A_dense = torch.empty((M, N), device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
        mask = torch.rand((M, N), device="cuda") > 0.65
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((M, K), device="cuda", dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        # power_of_2_64x32x16
        M, N, K = 64, 32, 16
        A_dense = torch.empty((M, N), device="cuda", dtype=dtype).uniform_(-3.0, 3.0)
        mask = torch.rand((M, N), device="cuda") > 0.70
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((M, K), device="cuda", dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        # non_power_of_2_negative_values
        M, N, K = 30, 50, 20
        A_dense = torch.empty((M, N), device="cuda", dtype=dtype).uniform_(-5.0, 5.0)
        mask = torch.rand((M, N), device="cuda") > 0.65
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device="cuda", dtype=dtype).uniform_(-3.0, 3.0),
                "C": torch.empty((M, K), device="cuda", dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        # non_power_of_2_255x100x33
        M, N, K = 255, 100, 33
        A_dense = torch.empty((M, N), device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
        mask = torch.rand((M, N), device="cuda") > 0.70
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((M, K), device="cuda", dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        # realistic_1000x500x64
        M, N, K = 1000, 500, 64
        A_dense = torch.empty((M, N), device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        mask = torch.rand((M, N), device="cuda") > 0.65
        A_sparse = A_dense * mask
        tests.append(
            {
                "A": A_sparse,
                "B": torch.empty((N, K), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty((M, K), device="cuda", dtype=dtype),
                "M": M,
                "N": N,
                "K": K,
                "nnz": int(mask.sum().item()),
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M = 4096
        N = 2048
        K = 512
        A_dense = torch.empty((M, N), device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        mask = torch.rand((M, N), device="cuda") > 0.65
        A_sparse = A_dense * mask
        nnz = int(mask.sum().item())
        B = torch.empty((N, K), device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        C = torch.empty((M, K), device="cuda", dtype=dtype)
        return {
            "A": A_sparse,
            "B": B,
            "C": C,
            "M": M,
            "N": N,
            "K": K,
            "nnz": nnz,
        }
