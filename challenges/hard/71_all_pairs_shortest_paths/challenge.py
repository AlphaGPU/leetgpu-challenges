import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


def _make_graph(N: int, density: float = 0.5, max_weight: float = 10.0, seed: int = None):
    """Create a random non-negative weighted directed graph as a flat float32 CUDA tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    d = torch.full((N * N,), float("inf"), device="cuda", dtype=torch.float32)
    d_view = d.view(N, N)
    d_view.fill_diagonal_(0.0)
    if N > 1:
        mask = torch.rand(N, N, device="cuda") < density
        mask.fill_diagonal_(False)
        weights = torch.rand(N, N, device="cuda") * max_weight + 0.1
        d_view[mask] = weights[mask]
    return d


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="All-Pairs Shortest Paths",
            atol=1e-02,
            rtol=1e-02,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(self, dist: torch.Tensor, output: torch.Tensor, N: int):
        assert dist.shape == (N * N,)
        assert output.shape == (N * N,)
        assert dist.dtype == output.dtype == torch.float32
        assert dist.device == output.device
        assert dist.device.type == "cuda"
        d = dist.view(N, N).clone()
        for k in range(N):
            d = torch.minimum(d, d[:, k : k + 1] + d[k : k + 1, :])
        output.copy_(d.view(-1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "dist": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        # 4-node directed graph: 0->1:5, 0->3:10, 1->2:3, 2->3:1
        # Shortest paths: 0->2 = 8 (via 1), 0->3 = 9 (via 1->2->3)
        inf = float("inf")
        dist = torch.tensor(
            [0.0, 5.0, inf, 10.0, inf, 0.0, 3.0, inf, inf, inf, 0.0, 1.0, inf, inf, inf, 0.0],
            device="cuda",
            dtype=torch.float32,
        )
        return {
            "dist": dist,
            "output": torch.empty(16, device="cuda", dtype=torch.float32),
            "N": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        inf = float("inf")

        def make_output(N):
            return torch.empty(N * N, device="cuda", dtype=torch.float32)

        # --- Edge cases ---

        # N=1: single vertex
        tests.append(
            {
                "dist": torch.tensor([0.0], device="cuda", dtype=torch.float32),
                "output": make_output(1),
                "N": 1,
            }
        )

        # N=2: disconnected graph (no edges between vertices)
        tests.append(
            {
                "dist": torch.tensor([0.0, inf, inf, 0.0], device="cuda", dtype=torch.float32),
                "output": make_output(2),
                "N": 2,
            }
        )

        # N=2: bidirectional edges
        tests.append(
            {
                "dist": torch.tensor([0.0, 3.0, 7.0, 0.0], device="cuda", dtype=torch.float32),
                "output": make_output(2),
                "N": 2,
            }
        )

        # N=3: chain 0->1->2; shortest path 0->2 = 2+3 = 5
        tests.append(
            {
                "dist": torch.tensor(
                    [0.0, 2.0, inf, inf, 0.0, 3.0, inf, inf, 0.0],
                    device="cuda",
                    dtype=torch.float32,
                ),
                "output": make_output(3),
                "N": 3,
            }
        )

        # N=4: graph with shortcut (same as example test)
        tests.append(
            {
                "dist": torch.tensor(
                    [
                        0.0,
                        5.0,
                        inf,
                        10.0,
                        inf,
                        0.0,
                        3.0,
                        inf,
                        inf,
                        inf,
                        0.0,
                        1.0,
                        inf,
                        inf,
                        inf,
                        0.0,
                    ],
                    device="cuda",
                    dtype=torch.float32,
                ),
                "output": make_output(4),
                "N": 4,
            }
        )

        # N=4: negative edge weights, no negative cycles (DAG: 0->1->2->3)
        # 0->1: -1, 1->2: 2, 2->3: -3, 0->3: 10
        # Shortest 0->2 = 1, 0->3 = -2, 1->3 = -1
        tests.append(
            {
                "dist": torch.tensor(
                    [
                        0.0,
                        -1.0,
                        inf,
                        10.0,
                        inf,
                        0.0,
                        2.0,
                        inf,
                        inf,
                        inf,
                        0.0,
                        -3.0,
                        inf,
                        inf,
                        inf,
                        0.0,
                    ],
                    device="cuda",
                    dtype=torch.float32,
                ),
                "output": make_output(4),
                "N": 4,
            }
        )

        # --- Power-of-2 sizes ---
        for N, seed in [(16, 1), (32, 2), (64, 3), (128, 4)]:
            tests.append(
                {
                    "dist": _make_graph(N, density=0.5, seed=seed),
                    "output": make_output(N),
                    "N": N,
                }
            )

        # --- Non-power-of-2 sizes ---
        for N, seed in [(30, 5), (100, 6), (255, 7)]:
            tests.append(
                {
                    "dist": _make_graph(N, density=0.4, seed=seed),
                    "output": make_output(N),
                    "N": N,
                }
            )

        # --- Realistic sizes ---
        for N, seed in [(512, 8)]:
            tests.append(
                {
                    "dist": _make_graph(N, density=0.3, seed=seed),
                    "output": make_output(N),
                    "N": N,
                }
            )

        # --- Special: all zero-weight edges (any path has cost 0) ---
        N = 8
        tests.append(
            {
                "dist": torch.zeros(N * N, device="cuda", dtype=torch.float32),
                "output": make_output(N),
                "N": N,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        N = 2048
        return {
            "dist": _make_graph(N, density=0.3, seed=42),
            "output": torch.empty(N * N, device="cuda", dtype=torch.float32),
            "N": N,
        }
