import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Grouped GEMM",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        group_offsets: torch.Tensor,
        C: torch.Tensor,
        G: int,
        M_total: int,
        K: int,
        N: int,
    ):
        assert A.shape == (M_total, K)
        assert B.shape == (G, K, N)
        assert group_offsets.shape == (G + 1,)
        assert C.shape == (M_total, N)
        assert A.dtype == torch.float32
        assert B.dtype == torch.float32
        assert C.dtype == torch.float32
        assert group_offsets.dtype == torch.int32
        assert A.device.type == "cuda"
        assert B.device.type == "cuda"
        assert group_offsets.device.type == "cuda"
        assert C.device.type == "cuda"

        offsets = group_offsets.tolist()
        for g in range(G):
            start = offsets[g]
            end = offsets[g + 1]
            if end > start:
                C[start:end] = torch.matmul(A[start:end], B[g])

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "group_offsets": (ctypes.POINTER(ctypes.c_int), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "out"),
            "G": (ctypes.c_int, "in"),
            "M_total": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def _build_test(
        self,
        group_sizes: List[int],
        K: int,
        N: int,
        seed: int = 0,
        zero_a: bool = False,
        all_negative: bool = False,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        G = len(group_sizes)
        M_total = sum(group_sizes)
        if zero_a:
            A = torch.zeros((max(M_total, 1), K), device="cuda", dtype=torch.float32)
        elif all_negative:
            A = -torch.empty((max(M_total, 1), K), device="cuda", dtype=torch.float32).uniform_(
                0.1, 1.0
            )
        else:
            A = torch.empty((max(M_total, 1), K), device="cuda", dtype=torch.float32).uniform_(
                -1.0, 1.0
            )
        if M_total == 0:
            A = A[:0]
        B = torch.empty((G, K, N), device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
        offsets = [0]
        for s in group_sizes:
            offsets.append(offsets[-1] + s)
        group_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        C = torch.empty((M_total, N), device="cuda", dtype=torch.float32)
        return {
            "A": A,
            "B": B,
            "group_offsets": group_offsets,
            "C": C,
            "G": G,
            "M_total": M_total,
            "K": K,
            "N": N,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        # G=2, M_total=4, K=2, N=3
        A = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            device="cuda",
            dtype=dtype,
        )
        B = torch.tensor(
            [
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            device="cuda",
            dtype=dtype,
        )
        group_offsets = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
        C = torch.empty((4, 3), device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "group_offsets": group_offsets,
            "C": C,
            "G": 2,
            "M_total": 4,
            "K": 2,
            "N": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Example case (G=2, very small)
        tests.append(self.generate_example_test())

        # Edge: single group, single row
        tests.append(self._build_test([1], K=4, N=4, seed=1))

        # Edge: single group, several rows (degenerate to one matmul)
        tests.append(self._build_test([8], K=16, N=16, seed=2))

        # Edge: zero input
        tests.append(self._build_test([3, 3], K=8, N=8, seed=3, zero_a=True))

        # Edge: multiple empty groups including at the boundary
        tests.append(self._build_test([0, 4, 0, 2, 0], K=8, N=8, seed=5))

        # Power-of-2: 4 even groups
        tests.append(self._build_test([16, 16, 16, 16], K=32, N=32, seed=6))

        # Power-of-2: 8 uneven groups (typical MoE imbalance)
        tests.append(self._build_test([4, 12, 20, 8, 32, 2, 16, 6], K=64, N=64, seed=7))

        # Non-power-of-2 dimensions
        tests.append(self._build_test([7, 11, 13], K=33, N=27, seed=8))

        # All-negative input
        tests.append(self._build_test([10, 20, 30], K=16, N=16, seed=9, all_negative=True))

        # Realistic MoE-style: 8 experts with imbalanced routing, hidden=128, intermediate=512
        torch.manual_seed(10)
        sizes = torch.randint(150, 350, (8,)).tolist()
        tests.append(self._build_test(sizes, K=128, N=512, seed=11))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(42)
        # MoE-style: 8 experts, 8192 total tokens, hidden=1024, intermediate=2048.
        # Token counts are unbalanced across experts to mirror real routing.
        G = 8
        M_total = 8192
        K = 1024
        N = 2048

        # Generate unbalanced group sizes that sum to M_total.
        weights = torch.empty(G).uniform_(0.5, 1.5)
        weights = weights / weights.sum()
        sizes = (weights * M_total).long()
        # Adjust last group to make sums match exactly.
        sizes[-1] += M_total - int(sizes.sum().item())
        offsets = torch.zeros(G + 1, dtype=torch.int32)
        offsets[1:] = torch.cumsum(sizes, dim=0).to(torch.int32)

        A = torch.empty((M_total, K), device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
        B = torch.empty((G, K, N), device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
        C = torch.empty((M_total, N), device="cuda", dtype=torch.float32)
        return {
            "A": A,
            "B": B,
            "group_offsets": offsets.cuda(),
            "C": C,
            "G": G,
            "M_total": M_total,
            "K": K,
            "N": N,
        }
