import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Stream Compaction",
            atol=0.0,
            rtol=0.0,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(self, A: torch.Tensor, N: int, out: torch.Tensor):
        assert A.shape == (N,), f"Expected A.shape=({N},), got {A.shape}"
        assert out.shape == (N,), f"Expected out.shape=({N},), got {out.shape}"
        assert A.dtype == torch.float32
        assert out.dtype == torch.float32
        assert A.device.type == "cuda"

        mask = A > 0
        selected = A[mask]
        k = selected.numel()
        out[:k].copy_(selected)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "N": (ctypes.c_int, "in"),
            "out": (ctypes.POINTER(ctypes.c_float), "out"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor([1.0, -2.0, 3.0, 0.0, -1.0, 4.0], device="cuda", dtype=dtype)
        N = 6
        out = torch.zeros(N, device="cuda", dtype=dtype)
        return {"A": A, "N": N, "out": out}

    def _make_test(self, N: int, lo: float = -2.0, hi: float = 2.0) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.empty(N, device="cuda", dtype=dtype).uniform_(lo, hi)
        out = torch.zeros(N, device="cuda", dtype=dtype)
        return {"A": A, "N": N, "out": out}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Edge cases — tiny sizes
        # N=1, zero (not positive, nothing selected)
        tests.append(
            {
                "A": torch.tensor([0.0], device="cuda", dtype=dtype),
                "N": 1,
                "out": torch.zeros(1, device="cuda", dtype=dtype),
            }
        )
        # N=1, positive (all selected)
        tests.append(
            {
                "A": torch.tensor([5.0], device="cuda", dtype=dtype),
                "N": 1,
                "out": torch.zeros(1, device="cuda", dtype=dtype),
            }
        )
        # N=4, mixed with exact zeros and negatives
        tests.append(
            {
                "A": torch.tensor([-1.0, 2.0, 0.0, 4.0], device="cuda", dtype=dtype),
                "N": 4,
                "out": torch.zeros(4, device="cuda", dtype=dtype),
            }
        )

        # Power-of-2 sizes
        # All positive — every element passes the predicate
        A_all_pos = torch.rand(16, device="cuda", dtype=dtype) + 0.1
        tests.append({"A": A_all_pos, "N": 16, "out": torch.zeros(16, device="cuda", dtype=dtype)})

        # All negative — no element passes the predicate
        A_all_neg = -(torch.rand(32, device="cuda", dtype=dtype) + 0.1)
        tests.append({"A": A_all_neg, "N": 32, "out": torch.zeros(32, device="cuda", dtype=dtype)})

        # Mixed, wide range
        tests.append(self._make_test(256, lo=-5.0, hi=5.0))
        tests.append(self._make_test(1024, lo=-10.0, hi=10.0))

        # Non-power-of-2
        tests.append(self._make_test(100, lo=-3.0, hi=3.0))
        tests.append(self._make_test(255, lo=-1.0, hi=1.0))

        # Realistic size
        tests.append(self._make_test(10000, lo=-100.0, hi=100.0))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50_000_000
        A = torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        out = torch.zeros(N, device="cuda", dtype=dtype)
        return {"A": A, "N": N, "out": out}
