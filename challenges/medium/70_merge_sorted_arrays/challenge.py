import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Merge Sorted Arrays",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int):
        assert A.shape == (M,)
        assert B.shape == (N,)
        assert C.shape == (M + N,)
        assert A.dtype == torch.float32
        assert B.dtype == torch.float32
        assert C.dtype == torch.float32
        assert A.device.type == "cuda"
        assert B.device.type == "cuda"
        assert C.device.type == "cuda"
        result = torch.sort(torch.cat([A, B]))[0]
        C.copy_(result)

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
        C = torch.empty(8, device="cuda", dtype=dtype)
        return {"A": A, "B": B, "C": C, "M": 4, "N": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        def make_sorted(size, low=-10.0, high=10.0):
            return torch.sort(torch.empty(size, device="cuda", dtype=dtype).uniform_(low, high))[0]

        # edge: M=1, N=1
        tests.append(
            {
                "A": torch.tensor([2.0], device="cuda", dtype=dtype),
                "B": torch.tensor([1.0], device="cuda", dtype=dtype),
                "C": torch.empty(2, device="cuda", dtype=dtype),
                "M": 1,
                "N": 1,
            }
        )

        # edge: single vs three, B entirely below A
        tests.append(
            {
                "A": torch.tensor([5.0], device="cuda", dtype=dtype),
                "B": torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype),
                "C": torch.empty(4, device="cuda", dtype=dtype),
                "M": 1,
                "N": 3,
            }
        )

        # edge: three vs one, A entirely below B
        tests.append(
            {
                "A": torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype),
                "B": torch.tensor([4.0], device="cuda", dtype=dtype),
                "C": torch.empty(4, device="cuda", dtype=dtype),
                "M": 3,
                "N": 1,
            }
        )

        # edge: all zeros (zero inputs)
        tests.append(
            {
                "A": torch.zeros(4, device="cuda", dtype=dtype),
                "B": torch.zeros(4, device="cuda", dtype=dtype),
                "C": torch.empty(8, device="cuda", dtype=dtype),
                "M": 4,
                "N": 4,
            }
        )

        # power-of-2: M=16, N=16, interleaved ranges
        A16 = make_sorted(16, 0.0, 1.0)
        B16 = make_sorted(16, 0.5, 1.5)
        tests.append(
            {"A": A16, "B": B16, "C": torch.empty(32, device="cuda", dtype=dtype), "M": 16, "N": 16}
        )

        # power-of-2: M=64, N=128
        A64 = make_sorted(64, -5.0, 5.0)
        B128 = make_sorted(128, -5.0, 5.0)
        tests.append(
            {
                "A": A64,
                "B": B128,
                "C": torch.empty(192, device="cuda", dtype=dtype),
                "M": 64,
                "N": 128,
            }
        )

        # power-of-2: M=512, N=256, all-negative inputs
        A512 = make_sorted(512, -100.0, -1.0)
        B256 = make_sorted(256, -50.0, -0.5)
        tests.append(
            {
                "A": A512,
                "B": B256,
                "C": torch.empty(768, device="cuda", dtype=dtype),
                "M": 512,
                "N": 256,
            }
        )

        # non-power-of-2: M=30, N=25, mixed values
        A30 = make_sorted(30, -10.0, 10.0)
        B25 = make_sorted(25, -10.0, 10.0)
        tests.append(
            {
                "A": A30,
                "B": B25,
                "C": torch.empty(55, device="cuda", dtype=dtype),
                "M": 30,
                "N": 25,
            }
        )

        # non-power-of-2: M=100, N=73
        A100 = make_sorted(100, -1.0, 1.0)
        B73 = make_sorted(73, -1.0, 1.0)
        tests.append(
            {
                "A": A100,
                "B": B73,
                "C": torch.empty(173, device="cuda", dtype=dtype),
                "M": 100,
                "N": 73,
            }
        )

        # non-power-of-2: M=255, N=200
        A255 = make_sorted(255, -100.0, 100.0)
        B200 = make_sorted(200, -100.0, 100.0)
        tests.append(
            {
                "A": A255,
                "B": B200,
                "C": torch.empty(455, device="cuda", dtype=dtype),
                "M": 255,
                "N": 200,
            }
        )

        # non-overlapping ranges (A fully below B — concat is already sorted)
        A_low = make_sorted(64, -10.0, 0.0)
        B_high = make_sorted(64, 1.0, 10.0)
        tests.append(
            {
                "A": A_low,
                "B": B_high,
                "C": torch.empty(128, device="cuda", dtype=dtype),
                "M": 64,
                "N": 64,
            }
        )

        # realistic: M=1000, N=500
        A1000 = make_sorted(1000, -50.0, 50.0)
        B500 = make_sorted(500, -50.0, 50.0)
        tests.append(
            {
                "A": A1000,
                "B": B500,
                "C": torch.empty(1500, device="cuda", dtype=dtype),
                "M": 1000,
                "N": 500,
            }
        )

        # realistic: M=5000, N=3000
        A5000 = make_sorted(5000, -1000.0, 1000.0)
        B3000 = make_sorted(3000, -1000.0, 1000.0)
        tests.append(
            {
                "A": A5000,
                "B": B3000,
                "C": torch.empty(8000, device="cuda", dtype=dtype),
                "M": 5000,
                "N": 3000,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M = 10_000_000
        N = 10_000_000
        A = torch.sort(torch.empty(M, device="cuda", dtype=dtype).uniform_(-1e6, 1e6))[0]
        B = torch.sort(torch.empty(N, device="cuda", dtype=dtype).uniform_(-1e6, 1e6))[0]
        C = torch.empty(M + N, device="cuda", dtype=dtype)
        return {"A": A, "B": B, "C": C, "M": M, "N": N}
