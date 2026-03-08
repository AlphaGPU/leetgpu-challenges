import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

_EPS = 1e-3


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="N-body Gravitational Force",
            atol=1e-2,
            rtol=1e-2,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        forces: torch.Tensor,
        N: int,
    ):
        assert positions.shape == (
            N,
            3,
        ), f"Expected positions.shape=({N}, 3), got {positions.shape}"
        assert masses.shape == (N,), f"Expected masses.shape=({N},), got {masses.shape}"
        assert forces.shape == (N, 3), f"Expected forces.shape=({N}, 3), got {forces.shape}"
        assert positions.dtype == torch.float32
        assert masses.dtype == torch.float32
        assert forces.dtype == torch.float32
        assert positions.device.type == "cuda"

        CHUNK = 1024
        result = torch.zeros((N, 3), device=positions.device, dtype=positions.dtype)
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            # r[i, k] = positions[start+k] - positions[i]: vector from particle i to source k
            # positions[start:end].unsqueeze(0): [1, C, 3] (C = end-start)
            # positions.unsqueeze(1):            [N, 1, 3]
            # broadcast result:                 [N, C, 3]
            r = positions[start:end].unsqueeze(0) - positions.unsqueeze(1)  # [N, C, 3]
            dist_sq = (r * r).sum(dim=2)  # [N, C]
            denom = (dist_sq + _EPS * _EPS) ** 1.5  # [N, C]
            mass_chunk = masses[start:end]  # [C]
            result += (mass_chunk.view(1, -1, 1) * r / denom.unsqueeze(2)).sum(dim=1)
        forces.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "positions": (ctypes.POINTER(ctypes.c_float), "in"),
            "masses": (ctypes.POINTER(ctypes.c_float), "in"),
            "forces": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 3.0, 4.0]], device="cuda", dtype=dtype)
        masses = torch.tensor([2.0, 1.0], device="cuda", dtype=dtype)
        forces = torch.zeros((2, 3), device="cuda", dtype=dtype)
        return {"positions": positions, "masses": masses, "forces": forces, "N": 2}

    def _make_test(self, N: int, pos_range: float = 5.0) -> Dict[str, Any]:
        dtype = torch.float32
        positions = torch.empty((N, 3), device="cuda", dtype=dtype).uniform_(-pos_range, pos_range)
        masses = torch.empty(N, device="cuda", dtype=dtype).uniform_(0.5, 2.0)
        forces = torch.zeros((N, 3), device="cuda", dtype=dtype)
        return {"positions": positions, "masses": masses, "forces": forces, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # N=1: single particle — no other bodies, forces must be zero
        tests.append(
            {
                "positions": torch.tensor([[1.0, 2.0, 3.0]], device="cuda", dtype=dtype),
                "masses": torch.tensor([1.5], device="cuda", dtype=dtype),
                "forces": torch.zeros((1, 3), device="cuda", dtype=dtype),
                "N": 1,
            }
        )

        # N=2: one particle at a negative position
        tests.append(
            {
                "positions": torch.tensor(
                    [[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device="cuda", dtype=dtype
                ),
                "masses": torch.tensor([1.0, 2.0], device="cuda", dtype=dtype),
                "forces": torch.zeros((2, 3), device="cuda", dtype=dtype),
                "N": 2,
            }
        )

        # N=4: all particles co-located at the origin — zero displacement gives zero forces
        tests.append(
            {
                "positions": torch.zeros((4, 3), device="cuda", dtype=dtype),
                "masses": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype),
                "forces": torch.zeros((4, 3), device="cuda", dtype=dtype),
                "N": 4,
            }
        )

        # Power-of-2 sizes
        tests.append(self._make_test(16))
        tests.append(self._make_test(256))

        # Non-power-of-2 sizes
        tests.append(self._make_test(30))
        tests.append(self._make_test(100))
        tests.append(self._make_test(255))

        # Realistic size
        tests.append(self._make_test(1024))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        N = 8192
        dtype = torch.float32
        positions = torch.empty((N, 3), device="cuda", dtype=dtype).uniform_(-10.0, 10.0)
        masses = torch.empty(N, device="cuda", dtype=dtype).uniform_(0.1, 10.0)
        forces = torch.zeros((N, 3), device="cuda", dtype=dtype)
        return {"positions": positions, "masses": masses, "forces": forces, "N": N}
