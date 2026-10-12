import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Top K Selection"
    atol = 1e-05
    rtol = 0.0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, k: int):
        assert input.shape == (N,)
        assert output.shape == (k,)
        assert input.dtype == output.dtype == torch.float32
        assert input.device == output.device
        topk = torch.topk(input, k, largest=True).values
        output.copy_(topk)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "k": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], device=self.device, dtype=dtype)
        output = torch.empty(3, device=self.device, dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "k": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append(
            {
                "input": torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], device=self.device, dtype=dtype),
                "output": torch.empty(3, device=self.device, dtype=dtype),
                "N": 5,
                "k": 3,
            }
        )
        # negative_numbers
        tests.append(
            {
                "input": torch.tensor(
                    [-2.0, -1.0, -3.0, -4.0, -5.0, -6.0], device=self.device, dtype=dtype
                ),
                "output": torch.empty(2, device=self.device, dtype=dtype),
                "N": 6,
                "k": 2,
            }
        )
        # all_equal
        tests.append(
            {
                "input": torch.tensor([7.0, 7.0, 7.0, 7.0], device=self.device, dtype=dtype),
                "output": torch.empty(3, device=self.device, dtype=dtype),
                "N": 4,
                "k": 3,
            }
        )
        # all_zeros with k == N
        tests.append(
            {
                "input": torch.zeros(4, device=self.device, dtype=dtype),
                "output": torch.empty(4, device=self.device, dtype=dtype),
                "N": 4,
                "k": 4,
            }
        )
        # single_element
        tests.append(
            {
                "input": torch.tensor([42.0], device=self.device, dtype=dtype),
                "output": torch.empty(1, device=self.device, dtype=dtype),
                "N": 1,
                "k": 1,
            }
        )
        # reverse_sorted
        tests.append(
            {
                "input": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device=self.device, dtype=dtype),
                "output": torch.empty(2, device=self.device, dtype=dtype),
                "N": 5,
                "k": 2,
            }
        )
        # large_random (simulated; actual is random in runner)
        N, k = 1000, 10
        tests.append(
            {
                "input": torch.empty(N, device=self.device, dtype=dtype).uniform_(-1000.0, 1000.0),
                "output": torch.empty(k, device=self.device, dtype=dtype),
                "N": N,
                "k": k,
            }
        )
        # Multiple winners share a strided partition, or a contiguous region.
        # Keeping just one maximum per partition must not discard other winners.
        for N, k, stride, start in [
            (65536, 50, 1024, 0),
            (1048576, 100, 4096, 0),
            (50000000, 100, 1, 25000000),
        ]:
            input = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
            # Integer-valued winners are exact in float32 and separated by much
            # more than the absolute comparison tolerance of 1e-5.
            input[start : start + k * stride : stride] = torch.arange(
                k + 100, 100, -1, device=self.device, dtype=dtype
            )
            tests.append(
                {
                    "input": input,
                    "output": torch.empty(k, device=self.device, dtype=dtype),
                    "N": N,
                    "k": k,
                }
            )
        # Ties and duplicates at k=100: the k-th value is repeated many times,
        # so the answer needs every copy of each tied winner up to exactly k.
        N, k = 1048576, 100
        input = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        # 40 copies of 300, 40 copies of 200, then 200 copies of 150 straddling
        # the k boundary, all scattered across the array.
        vals = torch.cat(
            [
                torch.full((40,), 300.0, device=self.device, dtype=dtype),
                torch.full((40,), 200.0, device=self.device, dtype=dtype),
                torch.full((200,), 150.0, device=self.device, dtype=dtype),
            ]
        )
        g = torch.Generator(device="cpu").manual_seed(29)
        pos = torch.randperm(N, generator=g)[: vals.numel()].to(self.device)
        input[pos] = vals
        tests.append(
            {
                "input": input,
                "output": torch.empty(k, device=self.device, dtype=dtype),
                "N": N,
                "k": k,
            }
        )
        # k=100 winners packed into one contiguous block at the very end of a
        # large non-power-of-two input, with duplicate values among them.
        N, k = 10000019, 100
        input = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1.0, 1.0)
        input[N - k :] = torch.arange(k, device=self.device, dtype=dtype).div(2).floor() + 10.0
        tests.append(
            {
                "input": input,
                "output": torch.empty(k, device=self.device, dtype=dtype),
                "N": N,
                "k": k,
            }
        )
        # Many duplicates of the global maximum (more than k copies) plus
        # all-negative background.
        N, k = 4194304, 100
        input = torch.empty(N, device=self.device, dtype=dtype).uniform_(-1000.0, -1.0)
        pos = torch.randperm(N, generator=g)[:150].to(self.device)
        input[pos] = 5.0
        tests.append(
            {
                "input": input,
                "output": torch.empty(k, device=self.device, dtype=dtype),
                "N": N,
                "k": k,
            }
        )
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 50000000
        k = 100
        return {
            "input": torch.empty(N, device=self.device, dtype=dtype).uniform_(-1e6, 1e6),
            "output": torch.empty(k, device=self.device, dtype=dtype),
            "N": N,
            "k": k,
        }
