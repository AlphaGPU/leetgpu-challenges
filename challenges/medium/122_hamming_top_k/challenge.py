import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

W = 8


class Challenge(ChallengeBase):
    name = "Hamming Top-K Search"
    atol = 0
    rtol = 0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        codes: torch.Tensor,
        query: torch.Tensor,
        output: torch.Tensor,
        N: int,
        k: int,
    ):
        assert codes.shape == (W * N,)
        assert query.shape == (W,)
        assert output.shape == (k,)
        assert codes.dtype == query.dtype == output.dtype == torch.int32
        assert 1 <= k <= N

        diff = codes.view(N, W) ^ query
        accumulator = torch.zeros_like(diff)

        for i in range(32):
            accumulator += (diff >> i) & 1

        dist = accumulator.sum(dim=1)
        score = dist * N + torch.arange(N, device=dist.device)
        _, index = torch.topk(score, k, largest=False, sorted=True)
        output.copy_(index.to(torch.int32))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "codes": (ctypes.POINTER(ctypes.c_int32), "in"),
            "query": (ctypes.POINTER(ctypes.c_int32), "in"),
            "output": (ctypes.POINTER(ctypes.c_int32), "out"),
            "N": (ctypes.c_int, "in"),
            "k": (ctypes.c_int, "in"),
        }

    def _case(self, codes: List[List[int]], query: List[int], k: int) -> Dict[str, Any]:
        flat = [x for code in codes for x in code]
        return {
            "codes": torch.tensor(flat, device=self.device, dtype=torch.int32),
            "query": torch.tensor(query, device=self.device, dtype=torch.int32),
            "output": torch.zeros(k, device=self.device, dtype=torch.int32),
            "N": len(codes),
            "k": k,
        }

    def _random_case(self, n: int, k: int) -> Dict[str, Any]:
        lo, hi = -(2**31), 2**31
        return {
            "codes": torch.randint(lo, hi, (W * n,), device=self.device, dtype=torch.int32),
            "query": torch.randint(lo, hi, (W,), device=self.device, dtype=torch.int32),
            "output": torch.zeros(k, device=self.device, dtype=torch.int32),
            "N": n,
            "k": k,
        }

    def _from_distances(self, dists: List[int], query: List[int], k: int) -> Dict[str, Any]:
        # Code i differs from the query in exactly dists[i] bits (the lowest ones of the 256).
        def word(q: int, mask: int, w: int) -> int:
            x = (q & 0xFFFFFFFF) ^ ((mask >> (32 * w)) & 0xFFFFFFFF)
            return x - (1 << 32) if x >= (1 << 31) else x

        codes = [[word(query[w], (1 << d) - 1, w) for w in range(W)] for d in dists]
        return self._case(codes, query, k)

    def generate_example_test(self) -> Dict[str, Any]:
        return self._case(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],
                [0, 2, 3, 0, 0, 0, 0, 0],
                [7, 0, 3, 0, 0, 0, 0, 0],
                [1, 3, 2, 0, 0, 0, 0, 0],
                [1, 2, 2, 0, 0, 0, 0, 0],
            ],
            [1, 2, 3, 0, 0, 0, 0, 0],
            2,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # tie at the cutoff, small codes
        tests.append(
            self._case(
                [
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                ],
                [0] * W,
                2,
            )
        )

        # k = 1, single code
        tests.append(self._random_case(1, 1))

        # k = N
        tests.append(self._random_case(16, 16))

        # all codes identical: every distance ties, answer is [0..k-1]
        tests.append(self._case([[5, -6, 7, -8, 9, -10, 11, -12]] * 100, [0] * W, 37))

        # bitwise NOT of the query has distance 256 and must come last; the query itself first
        q = [123456789, -987654321, 0, -1, -(2**31), 2**31 - 1, 1, -2]
        not_q = [~x for x in q]
        tests.append(self._case([not_q, q, [0] * W, not_q, q], q, 5))

        # sign bit and negative words: one bit per position across all 8 words
        codes = [[0] * W for _ in range(256)]
        for b in range(256):
            codes[b][b // 32] = 1 << (b % 32) if b % 32 != 31 else -(2**31)
        tests.append(self._case(codes, [0] * W, 20))

        # tie straddling the cutoff: distance 3 is partly in, partly out
        tests.append(
            self._from_distances(
                [3, 1, 3, 0, 3, 2, 3, 3, 1, 3, 256, 130, 33, 32, 31],
                [123456789, -987654321, 0, -1, -(2**31), 2**31 - 1, 1, -2],
                6,
            )
        )

        # power of 2
        tests.append(self._random_case(256, 16))

        # non power of 2
        tests.append(self._random_case(100, 10))

        # realistic
        tests.append(self._random_case(10000, 1024))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._random_case(4000000, 128)
