import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Hamming Distance"
    atol = 0
    rtol = 0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, codes: torch.Tensor, output: torch.Tensor, query: int, N: int):
        assert codes.shape == (N,)
        assert output.shape == (N,)
        assert codes.dtype == torch.int32
        assert output.dtype == torch.int32
        accumulator = torch.zeros_like(codes)
        codes = codes ^ query
        for i in range(32):
            accumulator += (codes >> i) & 1
        output.copy_(accumulator)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "codes": (ctypes.POINTER(ctypes.c_int32), "in"),
            "output": (ctypes.POINTER(ctypes.c_int32), "out"),
            "query": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def _case(self, codes: List[int], query: int) -> Dict[str, Any]:
        n = len(codes)
        return {
            "codes": torch.tensor(codes, device=self.device, dtype=torch.int32),
            "output": torch.zeros(n, device=self.device, dtype=torch.int32),
            "query": query,
            "N": n,
        }

    def _random_case(self, n: int, query: int) -> Dict[str, Any]:
        return {
            "codes": torch.randint(-(2**31), 2**31, (n,), device=self.device, dtype=torch.int32),
            "output": torch.zeros(n, device=self.device, dtype=torch.int32),
            "query": query,
            "N": n,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._case([5, 0, 7, -1], 5)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        test_cases = []

        # zeros
        test_cases.append(self._case([0], 0))

        # all bits set vs none
        test_cases.append(self._case([0, -1], 0))

        # negative query
        test_cases.append(self._case([5, 0, 7, -1], -1))

        # mixed values
        test_cases.append(self._case([0, 1, 11, -1], 0))

        # one bit per position, including the sign bit
        test_cases.append(self._case([1 << i for i in range(31)] + [-(2**31)], 0))

        # power of 2
        test_cases.append(self._random_case(256, 123456789))

        # non power of 2
        test_cases.append(self._random_case(100, -987654321))
        test_cases.append(self._random_case(255, 0))

        # realistic
        test_cases.append(self._random_case(10000, -2147483648))

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._random_case(25000000, 1431655765)
