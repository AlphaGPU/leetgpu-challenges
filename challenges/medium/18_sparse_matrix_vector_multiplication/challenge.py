import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Sparse Matrix-Vector Multiplication",
            atol=1e-03,
            rtol=1e-03,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
        # Accept A as either flattened (M*N,) or 2D (M, N)
        if A.shape == (M * N,):
            A_matrix = A.view(M, N)
        elif A.shape == (M, N):
            A_matrix = A
        else:
            raise AssertionError(f"A.shape {A.shape} does not match expected {(M*N,)} or {(M, N)}")
        assert x.shape == (N,)
        assert y.shape == (M,)
        result = torch.matmul(A_matrix, x)
        y.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "nnz": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor([
            5.0, 0.0, 0.0, 1.0,
            0.0, 2.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 4.0
        ], device="cuda", dtype=dtype)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        y = torch.empty(3, device="cuda", dtype=dtype)
        return {
            "A": A,
            "x": x,
            "y": y,
            "M": 3,
            "N": 4,
            "nnz": 5
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # small_test
        tests.append({
            "A": torch.tensor([[1.0, 2.0],[3.0, 4.0]], device="cuda", dtype=dtype),
            "x": torch.tensor([1.0, 1.0], device="cuda", dtype=dtype),
            "y": torch.empty(2, device="cuda", dtype=dtype),
            "M": 2,
            "N": 2,
            "nnz": 4
        })
        # identity_test
        tests.append({
            "A": torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]    
            ],device="cuda", dtype=dtype),
            "x": torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype),
            "y": torch.empty(3, device="cuda", dtype=dtype),
            "M": 3,
            "N": 3,
            "nnz": 3
        })
        # zero_test
        tests.append({
            "A": torch.zeros((2, 3), device="cuda", dtype=dtype),
            "x": torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype),
            "y": torch.empty(2, device="cuda", dtype=dtype),
            "M": 2,
            "N": 3,
            "nnz": 0
        })
        # single_element_per_row
        tests.append({
            "A": torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0]
            ], device="cuda", dtype=dtype),
            "x": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype),
            "y": torch.empty(3, device="cuda", dtype=dtype),
            "M": 3,
            "N": 4,
            "nnz": 3
        })
        # negative_values
        tests.append({
            "A": torch.tensor([
                [-1.0, -2.0, -3.0], 
                [-4.0, -5.0, -6.0]
            ], device="cuda", dtype=dtype),
            "x": torch.tensor([-1.0, -2.0, -3.0], device="cuda", dtype=dtype),
            "y": torch.empty(2, device="cuda", dtype=dtype),
            "M": 2,
            "N": 3,
            "nnz": 6
        })
        # medium_matrix
        tests.append({
            "A": torch.tensor([
                1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0,
                0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 7.0, 0.0,
                8.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0,
                0.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0,
                6.0, 0.0, 0.0, 7.0, 0.0, 8.0, 0.0, 0.0,
                0.0, 0.0, 9.0, 0.0, 1.0, 0.0, 2.0, 0.0,
                3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0,
                0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 9.0, 0.0,
                1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0
            ], device="cuda", dtype=dtype),
            "x": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "y": torch.empty(10, device="cuda", dtype=dtype),
            "M": 10,
            "N": 8,
            "nnz": 35
        })


        # random_sparse_matrix
        M_sparse = 20
        N_sparse = 20
        sparsity = 0.65

        # Generate random sparse matrix
        A_dense = torch.empty((M_sparse, N_sparse), device="cuda", dtype=dtype).uniform_(-5.0, 5.0)
        mask = torch.rand((M_sparse, N_sparse), device="cuda") > sparsity
        A_sparse = A_dense * mask
        nnz_sparse = int(mask.sum().item())

        tests.append({
            "A": A_sparse,
            "x": torch.empty(N_sparse, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
            "y": torch.zeros(M_sparse, device="cuda", dtype=dtype),
            "M": M_sparse,
            "N": N_sparse,
            "nnz": nnz_sparse
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M = 1000
        N = 10000
        nnz = 3500000
        A = torch.zeros((M, N), device="cuda", dtype=dtype)
        total_elements = M * N
        flat_indices = torch.randperm(total_elements, device="cuda")[:nnz]
        values = torch.empty(nnz, device="cuda", dtype=dtype).uniform_(-10.0, 10.0)
        A.view(-1)[flat_indices] = values
        
        # Create a mask: 35% entries will be kept, 65% set to zero
        x = torch.empty(N, device="cuda", dtype=dtype).uniform_(-5.0, 5.0)
        y = torch.empty(M, device="cuda", dtype=dtype)
        return {
            "A": A,
            "x": x,
            "y": y,
            "M": M,
            "N": N,
            "nnz": nnz
        } 
   