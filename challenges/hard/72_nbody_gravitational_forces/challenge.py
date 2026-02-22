import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

EPS = 1e-9


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="N-body Gravitational Forces",
            atol=0.05,
            rtol=1e-3,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        accelerations: torch.Tensor,
        N: int,
    ):
        N = int(N)
        assert positions.shape == (N, 3)
        assert masses.shape == (N,)
        assert accelerations.shape == (N, 3)
        assert positions.dtype == torch.float32
        assert masses.dtype == torch.float32
        assert accelerations.dtype == torch.float32
        assert positions.device.type == "cuda"

        result = torch.zeros((N, 3), dtype=torch.float32, device=positions.device)
        chunk = 512
        for j_start in range(0, N, chunk):
            j_end = min(j_start + chunk, N)
            pos_j = positions[j_start:j_end]  # (c, 3)
            mass_j = masses[j_start:j_end]  # (c,)
            # delta[i, k] = pos_j[k] - positions[i]
            delta = pos_j.unsqueeze(0) - positions.unsqueeze(1)  # (N, c, 3)
            dist_sq = (delta**2).sum(dim=-1, keepdim=True) + EPS  # (N, c, 1)
            dist_cubed = dist_sq**1.5  # (N, c, 1)
            contributions = mass_j.view(1, -1, 1) * delta / dist_cubed  # (N, c, 3)
            result += contributions.sum(dim=1)

        accelerations.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "positions": (ctypes.POINTER(ctypes.c_float), "in"),
            "masses": (ctypes.POINTER(ctypes.c_float), "in"),
            "accelerations": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"
        N = 4
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device,
        )
        masses = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device)
        accelerations = torch.empty((N, 3), dtype=dtype, device=device)
        return {
            "positions": positions,
            "masses": masses,
            "accelerations": accelerations,
            "N": N,
        }

    def _make_test(self, N: int, pos_scale: float = 1.0, seed: int = 0) -> Dict[str, Any]:
        dtype = torch.float32
        device = "cuda"
        gen = torch.Generator()
        gen.manual_seed(seed)
        positions = torch.rand(N, 3, dtype=dtype, generator=gen) * pos_scale
        # Normalize masses by N so total acceleration magnitude stays O(1)
        masses = (torch.rand(N, dtype=dtype, generator=gen) * 0.9 + 0.1) / N
        accelerations = torch.empty((N, 3), dtype=dtype, device=device)
        return {
            "positions": positions.to(device),
            "masses": masses.to(device),
            "accelerations": accelerations,
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # single particle — acceleration must be exactly zero
        tests.append(
            {
                "positions": torch.tensor([[3.0, 1.0, 4.0]], dtype=torch.float32, device="cuda"),
                "masses": torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                "accelerations": torch.empty((1, 3), dtype=torch.float32, device="cuda"),
                "N": 1,
            }
        )

        # two particles — equal and opposite accelerations (Newton's third law)
        tests.append(
            {
                "positions": torch.tensor(
                    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"
                ),
                "masses": torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda"),
                "accelerations": torch.empty((2, 3), dtype=torch.float32, device="cuda"),
                "N": 2,
            }
        )

        # 3 particles, example from problem description
        tests.append(
            {
                "positions": torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                    device="cuda",
                ),
                "masses": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"),
                "accelerations": torch.empty((3, 3), dtype=torch.float32, device="cuda"),
                "N": 3,
            }
        )

        # 4 particles (example test)
        tests.append(self._make_test(4, pos_scale=1.0, seed=1))

        # power-of-2 small: N=16
        tests.append(self._make_test(16, pos_scale=1.0, seed=2))

        # power-of-2: N=64
        tests.append(self._make_test(64, pos_scale=1.0, seed=3))

        # power-of-2: N=256
        tests.append(self._make_test(256, pos_scale=1.0, seed=4))

        # non-power-of-2: N=30
        tests.append(self._make_test(30, pos_scale=1.0, seed=5))

        # non-power-of-2: N=100
        tests.append(self._make_test(100, pos_scale=1.0, seed=6))

        # non-power-of-2: N=500
        tests.append(self._make_test(500, pos_scale=1.0, seed=7))

        # realistic: N=1024
        tests.append(self._make_test(1024, pos_scale=1.0, seed=8))

        # realistic: N=2048
        tests.append(self._make_test(2048, pos_scale=1.0, seed=9))

        # non-power-of-2 larger: N=3000
        tests.append(self._make_test(3000, pos_scale=1.0, seed=10))

        # larger power-of-2: N=4096
        tests.append(self._make_test(4096, pos_scale=1.0, seed=11))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._make_test(16384, pos_scale=1.0, seed=42)
