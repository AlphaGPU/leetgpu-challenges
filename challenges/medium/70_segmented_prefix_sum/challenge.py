import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Segmented Exclusive Prefix Sum",
            atol=1e-03,
            rtol=1e-03,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        values: torch.Tensor,
        flags: torch.Tensor,
        output: torch.Tensor,
        N: int,
    ):
        assert values.shape == (N,)
        assert flags.shape == (N,)
        assert output.shape == (N,)
        assert values.dtype == torch.float32
        assert flags.dtype == torch.int32
        assert values.device.type == "cuda"

        # Global exclusive prefix sum (use float64 for accuracy in reference).
        excl = torch.empty(N, dtype=torch.float64, device="cuda")
        excl[0] = 0.0
        if N > 1:
            excl[1:] = torch.cumsum(values[:-1].double(), dim=0)

        # The exclusive prefix sum within each segment equals the global exclusive
        # prefix sum minus the global exclusive prefix sum at the segment start.
        # Use segment IDs (0-indexed) to index the per-segment offsets.
        seg_ids = torch.cumsum(flags.long(), dim=0) - 1  # segment index for each element
        seg_mask = flags.bool()
        # excl value at each segment start
        seg_start_excl = excl[seg_mask]  # shape: (num_segments,)
        # Broadcast segment start offset to every element in that segment
        per_elem_offset = seg_start_excl[seg_ids]

        output.copy_((excl - per_elem_offset).float())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "values": (ctypes.POINTER(ctypes.c_float), "in"),
            "flags": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype_f = torch.float32
        dtype_i = torch.int32
        # Three segments: [1,2,3], [4,5], [6]
        # exclusive prefix sums: [0,1,3], [0,4], [0]
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda", dtype=dtype_f)
        flags = torch.tensor([1, 0, 0, 1, 0, 1], device="cuda", dtype=dtype_i)
        output = torch.empty(6, device="cuda", dtype=dtype_f)
        return {
            "values": values,
            "flags": flags,
            "output": output,
            "N": 6,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype_f = torch.float32
        dtype_i = torch.int32
        tests = []

        def make_test(vals, segs):
            """vals: list of floats, segs: list of segment start indices"""
            N = len(vals)
            flags = torch.zeros(N, dtype=dtype_i)
            for s in segs:
                flags[s] = 1
            return {
                "values": torch.tensor(vals, device="cuda", dtype=dtype_f),
                "flags": flags.cuda(),
                "output": torch.empty(N, device="cuda", dtype=dtype_f),
                "N": N,
            }

        def make_random_test(N, avg_seg_len, seed=None):
            if seed is not None:
                torch.manual_seed(seed)
            vals = torch.empty(N, dtype=dtype_f).uniform_(-10.0, 10.0)
            flags = torch.zeros(N, dtype=dtype_i)
            flags[0] = 1
            i = avg_seg_len
            while i < N:
                flags[i] = 1
                i += max(1, int(torch.randint(1, 2 * avg_seg_len + 1, (1,)).item()))
            return {
                "values": vals.cuda(),
                "flags": flags.cuda(),
                "output": torch.empty(N, device="cuda", dtype=dtype_f),
                "N": N,
            }

        # Edge: single element, single segment
        tests.append(make_test([5.0], [0]))

        # Edge: two elements, one segment
        tests.append(make_test([3.0, 7.0], [0]))

        # Edge: two elements, two segments
        tests.append(make_test([3.0, 7.0], [0, 1]))

        # Edge: four elements, all in one segment
        tests.append(make_test([1.0, 2.0, 3.0, 4.0], [0]))

        # Four elements, each its own segment (all outputs = 0)
        tests.append(make_test([1.0, -2.0, 3.0, -4.0], [0, 1, 2, 3]))

        # Negative values in mixed segments: two segments of length 3
        tests.append(make_test([-1.0, -2.0, -3.0, 5.0, 6.0, -7.0], [0, 3]))

        # Power-of-2: N=16, two equal segments
        tests.append(make_test([float(i) for i in range(16)], [0, 8]))

        # Power-of-2: N=32, segments of length 4
        tests.append(make_test([1.0] * 32, list(range(0, 32, 4))))

        # Power-of-2: N=64, random segment lengths ~8
        tests.append(make_random_test(64, avg_seg_len=8, seed=42))

        # Power-of-2: N=128, random segment lengths ~16
        tests.append(make_random_test(128, avg_seg_len=16, seed=7))

        # Non-power-of-2: N=30, segments of length ~5
        tests.append(make_random_test(30, avg_seg_len=5, seed=13))

        # Non-power-of-2: N=100, small segments of length ~3
        tests.append(make_random_test(100, avg_seg_len=3, seed=99))

        # Non-power-of-2: N=255, segments spanning multiple warps
        tests.append(make_random_test(255, avg_seg_len=32, seed=17))

        # Realistic: N=1024, segments of length ~64
        tests.append(make_random_test(1024, avg_seg_len=64, seed=11))

        # Realistic: N=10000, segments crossing block boundaries
        tests.append(make_random_test(10000, avg_seg_len=256, seed=55))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype_f = torch.float32
        dtype_i = torch.int32
        N = 50_000_000
        torch.manual_seed(42)
        vals = torch.empty(N, dtype=dtype_f).uniform_(-1.0, 1.0)
        flags = torch.zeros(N, dtype=dtype_i)
        flags[0] = 1
        # Segments of average length 256 (crosses many thread blocks)
        seg_starts = torch.arange(256, N, 256, dtype=torch.long)
        flags[seg_starts] = 1
        return {
            "values": vals.cuda(),
            "flags": flags.cuda(),
            "output": torch.empty(N, device="cuda", dtype=dtype_f),
            "N": N,
        }
