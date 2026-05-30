import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="MoE Token Dispatch",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        x: torch.Tensor,
        expert_idx: torch.Tensor,
        dispatched_x: torch.Tensor,
        token_counts: torch.Tensor,
        T: int,
        D: int,
        E: int,
        capacity: int,
    ):
        assert x.shape == (T, D)
        assert expert_idx.shape == (T,)
        assert dispatched_x.shape == (E, capacity, D)
        assert token_counts.shape == (E,)
        assert x.dtype == torch.float32
        assert expert_idx.dtype == torch.int32
        assert dispatched_x.dtype == torch.float32
        assert token_counts.dtype == torch.int32
        assert x.device.type == "cuda"
        assert expert_idx.device.type == "cuda"

        for e in range(E):
            # torch.where returns indices in ascending order — stable within each expert
            indices = torch.where(expert_idx == e)[0]
            count = int(indices.shape[0])
            assert count <= capacity, f"Expert {e} has {count} tokens but capacity is {capacity}"
            dispatched_x[e, :count] = x[indices]
            token_counts[e] = count

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "expert_idx": (ctypes.POINTER(ctypes.c_int), "in"),
            "dispatched_x": (ctypes.POINTER(ctypes.c_float), "out"),
            "token_counts": (ctypes.POINTER(ctypes.c_int), "out"),
            "T": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "E": (ctypes.c_int, "in"),
            "capacity": (ctypes.c_int, "in"),
        }

    def _make_test(
        self,
        T: int,
        D: int,
        E: int,
        expert_idx_tensor: torch.Tensor = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        capacity = T
        x = torch.randn(T, D, device="cuda", dtype=torch.float32)
        if expert_idx_tensor is not None:
            expert_idx = expert_idx_tensor
        else:
            expert_idx = torch.randint(0, E, (T,), device="cuda", dtype=torch.int32)
        dispatched_x = torch.zeros(E, capacity, D, device="cuda", dtype=torch.float32)
        token_counts = torch.zeros(E, device="cuda", dtype=torch.int32)
        return {
            "x": x,
            "expert_idx": expert_idx,
            "dispatched_x": dispatched_x,
            "token_counts": token_counts,
            "T": T,
            "D": D,
            "E": E,
            "capacity": capacity,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        T, D, E = 4, 3, 2
        capacity = T
        x = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        expert_idx = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
        dispatched_x = torch.zeros(E, capacity, D, device="cuda", dtype=torch.float32)
        token_counts = torch.zeros(E, device="cuda", dtype=torch.int32)
        return {
            "x": x,
            "expert_idx": expert_idx,
            "dispatched_x": dispatched_x,
            "token_counts": token_counts,
            "T": T,
            "D": D,
            "E": E,
            "capacity": capacity,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge case: single token goes to expert 0, expert 1 is empty
        tests.append(
            self._make_test(
                1,
                4,
                2,
                expert_idx_tensor=torch.tensor([0], device="cuda", dtype=torch.int32),
            )
        )

        # Edge case: two tokens, both assigned to expert 0
        tests.append(
            self._make_test(
                2,
                4,
                2,
                expert_idx_tensor=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
            )
        )

        # Edge case: exactly one token per expert (T == E)
        tests.append(
            self._make_test(
                4,
                8,
                4,
                expert_idx_tensor=torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32),
                seed=1,
            )
        )

        # Skewed distribution: 6 of 8 tokens go to expert 0
        skewed = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2], device="cuda", dtype=torch.int32)
        tests.append(self._make_test(8, 8, 4, expert_idx_tensor=skewed, seed=10))

        # Power-of-2: T=32, cycling uniformly through 4 experts
        uniform32 = (torch.arange(32, device="cuda") % 4).to(torch.int32)
        tests.append(self._make_test(32, 16, 4, expert_idx_tensor=uniform32, seed=2))

        # Power-of-2: T=256, random assignments to 8 experts
        tests.append(self._make_test(256, 64, 8, seed=3))

        # Non-power-of-2: T=30, cycling uniformly
        uniform30 = (torch.arange(30, device="cuda") % 4).to(torch.int32)
        tests.append(self._make_test(30, 16, 4, expert_idx_tensor=uniform30, seed=4))

        # Non-power-of-2: T=100, random assignments to 6 experts (includes negatives in x)
        tests.append(self._make_test(100, 32, 6, seed=5))

        # Realistic: T=1024 tokens, D=128, E=8 (includes negatives in x)
        tests.append(self._make_test(1024, 128, 8, seed=6))

        # Zero x values, random routing
        torch.manual_seed(7)
        zero_x = torch.zeros(64, 32, device="cuda", dtype=torch.float32)
        tests.append(
            {
                "x": zero_x,
                "expert_idx": torch.randint(0, 4, (64,), device="cuda", dtype=torch.int32),
                "dispatched_x": torch.zeros(4, 64, 32, device="cuda", dtype=torch.float32),
                "token_counts": torch.zeros(4, device="cuda", dtype=torch.int32),
                "T": 64,
                "D": 32,
                "E": 4,
                "capacity": 64,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        T, D, E = 16384, 512, 8
        capacity = T
        torch.manual_seed(0)
        x = torch.randn(T, D, device="cuda", dtype=torch.float32)
        expert_idx = torch.randint(0, E, (T,), device="cuda", dtype=torch.int32)
        dispatched_x = torch.zeros(E, capacity, D, device="cuda", dtype=torch.float32)
        token_counts = torch.zeros(E, device="cuda", dtype=torch.int32)
        return {
            "x": x,
            "expert_idx": expert_idx,
            "dispatched_x": dispatched_x,
            "token_counts": token_counts,
            "T": T,
            "D": D,
            "E": E,
            "capacity": capacity,
        }
