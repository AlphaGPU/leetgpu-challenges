import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Connected Components",
            atol=0,
            rtol=0,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        labels: torch.Tensor,
        N: int,
        M: int,
    ):
        assert src.dtype == torch.int32
        assert dst.dtype == torch.int32
        assert labels.dtype == torch.int32
        assert src.shape == (M,)
        assert dst.shape == (M,)
        assert labels.shape == (N,)
        assert src.device.type == "cuda"
        assert dst.device.type == "cuda"
        assert labels.device.type == "cuda"

        # Label propagation: labels[i] converges to the minimum vertex ID
        # in the connected component containing vertex i.
        comp = torch.arange(N, dtype=torch.int32, device="cuda")

        if M > 0:
            # Build undirected edge list (each edge appears in both directions)
            src_long = src.long()
            dst_long = dst.long()
            edge_src = torch.cat([src_long, dst_long])
            edge_dst = torch.cat([dst_long, src_long])

            for _ in range(N):
                neighbor_labels = comp[edge_dst]
                new_comp = comp.clone()
                new_comp.scatter_reduce_(
                    0, edge_src, neighbor_labels, reduce="amin", include_self=True
                )
                if torch.equal(new_comp, comp):
                    break
                comp = new_comp

        labels.copy_(comp)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "src": (ctypes.POINTER(ctypes.c_int), "in"),
            "dst": (ctypes.POINTER(ctypes.c_int), "in"),
            "labels": (ctypes.POINTER(ctypes.c_int), "out"),
            "N": (ctypes.c_int, "in"),
            "M": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        # Graph: 5 vertices, 3 edges
        # Edges: (0,1), (1,2), (3,4)
        # Components: {0,1,2} -> label 0, {3,4} -> label 3
        N, M = 5, 3
        src = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
        dst = torch.tensor([1, 2, 4], dtype=torch.int32, device="cuda")
        labels = torch.zeros(N, dtype=torch.int32, device="cuda")
        return {"src": src, "dst": dst, "labels": labels, "N": N, "M": M}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Test 1: Single vertex, no edges
        tests.append(
            {
                "src": torch.zeros(1, dtype=torch.int32, device="cuda"),
                "dst": torch.zeros(1, dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(1, dtype=torch.int32, device="cuda"),
                "N": 1,
                "M": 1,
            }
        )

        # Test 2: Two vertices, one edge (single component)
        tests.append(
            {
                "src": torch.tensor([0], dtype=torch.int32, device="cuda"),
                "dst": torch.tensor([1], dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(2, dtype=torch.int32, device="cuda"),
                "N": 2,
                "M": 1,
            }
        )

        # Test 3: Four isolated vertices (no edges; each is its own component)
        tests.append(
            {
                "src": torch.tensor([0], dtype=torch.int32, device="cuda"),
                "dst": torch.tensor([0], dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(4, dtype=torch.int32, device="cuda"),
                "N": 4,
                "M": 1,
            }
        )

        # Test 4: Two disjoint pairs
        tests.append(
            {
                "src": torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
                "dst": torch.tensor([1, 3], dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(4, dtype=torch.int32, device="cuda"),
                "N": 4,
                "M": 2,
            }
        )

        # Test 5: Line graph 0-1-2-3-4-5-6-7 (chain, power-of-2 size)
        N, M = 8, 7
        tests.append(
            {
                "src": torch.arange(0, M, dtype=torch.int32, device="cuda"),
                "dst": torch.arange(1, N, dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(N, dtype=torch.int32, device="cuda"),
                "N": N,
                "M": M,
            }
        )

        # Test 6: Star graph — center=0, leaves=1..15 (power-of-2 N=16)
        N, M = 16, 15
        tests.append(
            {
                "src": torch.zeros(M, dtype=torch.int32, device="cuda"),
                "dst": torch.arange(1, N, dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(N, dtype=torch.int32, device="cuda"),
                "N": N,
                "M": M,
            }
        )

        # Test 7: 10x10 grid graph (N=100, non-power-of-2)
        N = 100
        edges = []
        for r in range(10):
            for c in range(10):
                v = r * 10 + c
                if c + 1 < 10:
                    edges.append((v, v + 1))
                if r + 1 < 10:
                    edges.append((v, v + 10))
        M = len(edges)
        src_list, dst_list = zip(*edges)
        tests.append(
            {
                "src": torch.tensor(src_list, dtype=torch.int32, device="cuda"),
                "dst": torch.tensor(dst_list, dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(N, dtype=torch.int32, device="cuda"),
                "N": N,
                "M": M,
            }
        )

        # Test 8: Three components of sizes 5, 3, 7 (N=15, non-power-of-2)
        N = 15
        src_list = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13]
        dst_list = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        M = len(src_list)
        tests.append(
            {
                "src": torch.tensor(src_list, dtype=torch.int32, device="cuda"),
                "dst": torch.tensor(dst_list, dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(N, dtype=torch.int32, device="cuda"),
                "N": N,
                "M": M,
            }
        )

        # Test 9: Random sparse graph, realistic size (N=1000, M=3000)
        torch.manual_seed(42)
        N, M = 1000, 3000
        tests.append(
            {
                "src": torch.randint(0, N, (M,), dtype=torch.int32, device="cuda"),
                "dst": torch.randint(0, N, (M,), dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(N, dtype=torch.int32, device="cuda"),
                "N": N,
                "M": M,
            }
        )

        # Test 10: Complete graph K4 (all vertices connected)
        N = 4
        src_list = [0, 0, 0, 1, 1, 2]
        dst_list = [1, 2, 3, 2, 3, 3]
        M = len(src_list)
        tests.append(
            {
                "src": torch.tensor(src_list, dtype=torch.int32, device="cuda"),
                "dst": torch.tensor(dst_list, dtype=torch.int32, device="cuda"),
                "labels": torch.zeros(N, dtype=torch.int32, device="cuda"),
                "N": N,
                "M": M,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(42)
        N = 1_000_000
        M = 5_000_000
        src = torch.randint(0, N, (M,), dtype=torch.int32, device="cuda")
        dst = torch.randint(0, N, (M,), dtype=torch.int32, device="cuda")
        labels = torch.zeros(N, dtype=torch.int32, device="cuda")
        return {"src": src, "dst": dst, "labels": labels, "N": N, "M": M}
