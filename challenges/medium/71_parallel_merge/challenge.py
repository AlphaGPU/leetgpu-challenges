import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Parallel Merge",
            atol=0.0,
            rtol=0.0,
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
    ):
        assert A.shape == (M,), f"Expected A.shape=({M},), got {A.shape}"
        assert B.shape == (N,), f"Expected B.shape=({N},), got {B.shape}"
        assert C.shape == (M + N,), f"Expected C.shape=({M + N},), got {C.shape}"
        assert A.dtype == torch.float32
        assert B.dtype == torch.float32
        assert C.dtype == torch.float32
        assert A.device.type == "cuda"

        merged, _ = torch.sort(torch.cat([A, B]))
        C.copy_(merged)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "A": (ctypes.POINTER(ctypes.c_float), "in"),
            "B": (ctypes.POINTER(ctypes.c_float), "in"),
            "C": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        A = torch.tensor([1.0, 3.0, 5.0, 7.0], device="cuda", dtype=dtype)
        B = torch.tensor([2.0, 4.0, 6.0, 8.0], device="cuda", dtype=dtype)
        M, N = 4, 4
        C = torch.empty(M + N, device="cuda", dtype=dtype)
        return {"A": A, "B": B, "C": C, "M": M, "N": N}

    def _make_test(self, M: int, N: int, lo: float = -10.0, hi: float = 10.0) -> Dict[str, Any]:
        dtype = torch.float32
        A, _ = torch.sort(torch.empty(M, device="cuda", dtype=dtype).uniform_(lo, hi))
        B, _ = torch.sort(torch.empty(N, device="cuda", dtype=dtype).uniform_(lo, hi))
        C = torch.empty(M + N, device="cuda", dtype=dtype)
        return {"A": A, "B": B, "C": C, "M": M, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Edge cases — tiny sizes
        tests.append(
            {
                "A": torch.tensor([0.0], device="cuda", dtype=dtype),
                "B": torch.tensor([1.0], device="cuda", dtype=dtype),
                "C": torch.empty(2, device="cuda", dtype=dtype),
                "M": 1,
                "N": 1,
            }
        )
        tests.append(
            {
                "A": torch.tensor([2.0], device="cuda", dtype=dtype),
                "B": torch.tensor([-1.0, 1.0, 3.0], device="cuda", dtype=dtype),
                "C": torch.empty(4, device="cuda", dtype=dtype),
                "M": 1,
                "N": 3,
            }
        )
        tests.append(
            {
                "A": torch.tensor([-1.0, 1.0, 3.0], device="cuda", dtype=dtype),
                "B": torch.tensor([2.0], device="cuda", dtype=dtype),
                "C": torch.empty(4, device="cuda", dtype=dtype),
                "M": 3,
                "N": 1,
            }
        )
        # All zeros
        tests.append(
            {
                "A": torch.zeros(2, device="cuda", dtype=dtype),
                "B": torch.zeros(2, device="cuda", dtype=dtype),
                "C": torch.empty(4, device="cuda", dtype=dtype),
                "M": 2,
                "N": 2,
            }
        )

        # Power-of-2 sizes
        tests.append(self._make_test(16, 16))
        tests.append(self._make_test(32, 32, lo=-100.0, hi=0.0))  # all negative
        tests.append(self._make_test(64, 128))
        tests.append(self._make_test(512, 512))
        tests.append(self._make_test(1024, 1024))

        # Non-power-of-2 sizes
        tests.append(self._make_test(30, 33))
        tests.append(self._make_test(100, 77))
        tests.append(self._make_test(255, 127))

        # A entirely less than B (no interleaving needed)
        A_low, _ = torch.sort(torch.empty(256, device="cuda", dtype=dtype).uniform_(-20.0, -10.0))
        B_high, _ = torch.sort(torch.empty(256, device="cuda", dtype=dtype).uniform_(10.0, 20.0))
        tests.append(
            {
                "A": A_low,
                "B": B_high,
                "C": torch.empty(512, device="cuda", dtype=dtype),
                "M": 256,
                "N": 256,
            }
        )

        # Many duplicate values
        A_dup = torch.sort(torch.randint(0, 5, (128,), device="cuda").to(dtype=dtype)).values
        B_dup = torch.sort(torch.randint(0, 5, (128,), device="cuda").to(dtype=dtype)).values
        tests.append(
            {
                "A": A_dup,
                "B": B_dup,
                "C": torch.empty(256, device="cuda", dtype=dtype),
                "M": 128,
                "N": 128,
            }
        )

        # Realistic size
        tests.append(self._make_test(5000, 7000))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M = 25_000_000
        N = 25_000_000
        A, _ = torch.sort(torch.empty(M, device="cuda", dtype=dtype).uniform_(-1.0, 1.0))
        B, _ = torch.sort(torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0))
        C = torch.empty(M + N, device="cuda", dtype=dtype)
        return {"A": A, "B": B, "C": C, "M": M, "N": N}
