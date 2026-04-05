import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

# OCP FP4 E2M1 lookup table: 4-bit unsigned index -> float value.
# Bit layout: [sign | exp1 exp0 | mantissa]. Sixteen representable values.
FP4_E2M1_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="FP4 MatMul",
            atol=5e-02,
            rtol=5e-02,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        x: torch.Tensor,
        w_q: torch.Tensor,
        scales: torch.Tensor,
        y: torch.Tensor,
        M: int,
        N: int,
        K: int,
        group_size: int,
    ):
        assert x.shape == (M, K)
        assert w_q.shape == (N, K // 2)
        assert scales.shape == (N, K // group_size)
        assert y.shape == (M, N)
        assert x.dtype == torch.float16
        assert w_q.dtype == torch.uint8
        assert scales.dtype == torch.float16
        assert y.dtype == torch.float16
        assert x.device.type == "cuda"
        assert w_q.device.type == "cuda"
        assert scales.device.type == "cuda"
        assert y.device.type == "cuda"

        # Decode packed FP4 E2M1 nibbles via lookup table.
        # w_q[n, i] holds two FP4 values: w[n, 2*i] in the high nibble (bits 7:4)
        # and w[n, 2*i+1] in the low nibble (bits 3:0).
        table = torch.tensor(FP4_E2M1_TABLE, device=x.device, dtype=torch.float32)
        high = ((w_q >> 4) & 0xF).to(torch.long)  # [N, K//2]
        low = (w_q & 0xF).to(torch.long)  # [N, K//2]
        w_high = table[high]  # [N, K//2]
        w_low = table[low]  # [N, K//2]
        w_fp4 = torch.stack([w_high, w_low], dim=-1).reshape(N, K)  # [N, K]

        # Apply group-wise FP16 scales: each contiguous block of `group_size`
        # weights along K shares one scale.
        n_groups = K // group_size
        w_groups = w_fp4.reshape(N, n_groups, group_size)
        scales_f = scales.float().unsqueeze(-1)  # [N, n_groups, 1]
        w_dequant = (w_groups * scales_f).reshape(N, K)

        y.copy_((x.float() @ w_dequant.T).half())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "w_q": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "scales": (ctypes.POINTER(ctypes.c_uint16), "in"),
            "y": (ctypes.POINTER(ctypes.c_uint16), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "group_size": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, M: int, N: int, K: int, group_size: int, zero_x: bool = False):
        device = "cuda"
        if zero_x:
            x = torch.zeros(M, K, device=device, dtype=torch.float16)
        else:
            x = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5
        w_q = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
        scales = torch.rand(N, K // group_size, device=device, dtype=torch.float16) * 0.1 + 0.01
        y = torch.empty(M, N, device=device, dtype=torch.float16)
        return {
            "x": x,
            "w_q": w_q,
            "scales": scales,
            "y": y,
            "M": M,
            "N": N,
            "K": K,
            "group_size": group_size,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        M, N, K, group_size = 2, 4, 4, 2

        x = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float16,
        )
        # Packed FP4 E2M1 weights (high nibble first).
        # Row 0: FP4 [1.0,1.0,1.0,1.0] -> nibbles [0x2,0x2,0x2,0x2] -> bytes [0x22,0x22] = [34,34]
        # Row 1: FP4 [2.0,2.0,2.0,2.0] -> nibbles [0x4,0x4,0x4,0x4] -> bytes [0x44,0x44] = [68,68]
        # Row 2: FP4 [-1,-1,-1,-1] -> nibbles [0xA,0xA,0xA,0xA] -> bytes [0xAA,0xAA] = [170,170]
        # Row 3: FP4 [0.0,0.0,0.0,0.0] -> nibbles [0x0,0x0,0x0,0x0] -> bytes [0x00,0x00] = [0,0]
        w_q = torch.tensor(
            [[34, 34], [68, 68], [170, 170], [0, 0]],
            dtype=torch.uint8,
            device=device,
        )
        scales = torch.full((N, K // group_size), 0.5, device=device, dtype=torch.float16)
        y = torch.empty(M, N, device=device, dtype=torch.float16)

        return {
            "x": x,
            "w_q": w_q,
            "scales": scales,
            "y": y,
            "M": M,
            "N": N,
            "K": K,
            "group_size": group_size,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge cases with tiny shapes.
        tests.append(self._make_test_case(1, 2, 4, 2, zero_x=True))
        tests.append(self._make_test_case(2, 4, 4, 2))
        tests.append(self._make_test_case(3, 5, 8, 4))

        # Power-of-2 shapes.
        tests.append(self._make_test_case(16, 16, 32, 16))
        tests.append(self._make_test_case(32, 64, 64, 32))
        tests.append(self._make_test_case(128, 128, 256, 32))

        # Non-power-of-2 shapes.
        tests.append(self._make_test_case(30, 50, 64, 32))
        tests.append(self._make_test_case(100, 200, 128, 32))
        tests.append(self._make_test_case(255, 100, 128, 32))

        # Realistic LLM inference shape.
        tests.append(self._make_test_case(512, 1024, 1024, 32))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # Matches the FP4 matmul shapes reported in AutoKernel community results.
        return self._make_test_case(2048, 8192, 3072, 32)
