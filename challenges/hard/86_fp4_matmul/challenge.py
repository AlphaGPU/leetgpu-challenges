import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

# OCP FP4 E2M1 lookup table: 4-bit unsigned index -> float value.
# Bit layout: [sign | exp1 exp0 | mantissa].
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

# NVFP4 block size along the reduction dimension. Each block of 16 FP4
# values shares one E4M3 scale. Matches CUTLASS / qutlass NVFP4 layout.
BLOCK_SIZE = 16


def _decode_fp4_packed(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Decode a (rows, cols/2) uint8 tensor of packed FP4 E2M1 nibbles into
    a (rows, cols) float32 tensor. High nibble stores the even-index value,
    low nibble stores the odd-index value."""
    table = torch.tensor(FP4_E2M1_TABLE, device=packed.device, dtype=torch.float32)
    high = ((packed >> 4) & 0xF).to(torch.long)
    low = (packed & 0xF).to(torch.long)
    decoded = torch.stack([table[high], table[low]], dim=-1).reshape(rows, cols)
    return decoded


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="NVFP4 Matrix Multiplication",
            atol=1e-01,
            rtol=5e-02,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        x_q: torch.Tensor,
        x_scales: torch.Tensor,
        w_q: torch.Tensor,
        w_scales: torch.Tensor,
        alpha: float,
        y: torch.Tensor,
        M: int,
        N: int,
        K: int,
    ):
        assert K % BLOCK_SIZE == 0, "K must be divisible by 16 (NVFP4 block size)"
        assert x_q.shape == (M, K // 2)
        assert x_scales.shape == (M, K // BLOCK_SIZE)
        assert w_q.shape == (N, K // 2)
        assert w_scales.shape == (N, K // BLOCK_SIZE)
        assert y.shape == (M, N)
        assert x_q.dtype == torch.uint8
        assert w_q.dtype == torch.uint8
        assert x_scales.dtype == torch.uint8
        assert w_scales.dtype == torch.uint8
        assert y.dtype == torch.float16
        assert x_q.device.type == "cuda"
        assert x_scales.device.type == "cuda"
        assert w_q.device.type == "cuda"
        assert w_scales.device.type == "cuda"
        assert y.device.type == "cuda"

        # Decode packed FP4 operands to float32.
        x_fp4 = _decode_fp4_packed(x_q, M, K)
        w_fp4 = _decode_fp4_packed(w_q, N, K)

        # Decode E4M3 per-block scales to float32.
        xs = x_scales.view(torch.float8_e4m3fn).float()  # (M, K/16)
        ws = w_scales.view(torch.float8_e4m3fn).float()  # (N, K/16)

        # Apply per-block scales along the reduction dimension.
        n_blocks = K // BLOCK_SIZE
        x_dq = (x_fp4.reshape(M, n_blocks, BLOCK_SIZE) * xs.unsqueeze(-1)).reshape(M, K)
        w_dq = (w_fp4.reshape(N, n_blocks, BLOCK_SIZE) * ws.unsqueeze(-1)).reshape(N, K)

        # NVFP4 matmul: y = alpha * (x @ w^T), result cast to FP16.
        out = float(alpha) * (x_dq @ w_dq.T)
        y.copy_(out.half())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x_q": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "x_scales": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "w_q": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "w_scales": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "alpha": (ctypes.c_float, "in"),
            "y": (ctypes.POINTER(ctypes.c_uint16), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, M: int, N: int, K: int, zero_x: bool = False, alpha: float = 1.0):
        assert K % BLOCK_SIZE == 0, "K must be divisible by 16"
        device = "cuda"
        if zero_x:
            x_q = torch.zeros(M, K // 2, dtype=torch.uint8, device=device)
        else:
            x_q = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
        w_q = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)

        # Positive E4M3 block scales in a representable range.
        x_scales_f = torch.rand(M, K // BLOCK_SIZE, device=device) * 1.5 + 0.5
        w_scales_f = torch.rand(N, K // BLOCK_SIZE, device=device) * 1.5 + 0.5
        x_scales = x_scales_f.to(torch.float8_e4m3fn).view(torch.uint8)
        w_scales = w_scales_f.to(torch.float8_e4m3fn).view(torch.uint8)

        y = torch.empty(M, N, device=device, dtype=torch.float16)
        return {
            "x_q": x_q,
            "x_scales": x_scales,
            "w_q": w_q,
            "w_scales": w_scales,
            "alpha": alpha,
            "y": y,
            "M": M,
            "N": N,
            "K": K,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = "cuda"
        M, N, K = 2, 2, 16

        # x row 0: sixteen FP4 values = 1.0 (nibble 0x2) -> 8 bytes of 0x22
        # x row 1: sixteen FP4 values = 0.5 (nibble 0x1) -> 8 bytes of 0x11
        x_q = torch.tensor(
            [[0x22] * 8, [0x11] * 8],
            dtype=torch.uint8,
            device=device,
        )
        # w row 0: sixteen FP4 values = 2.0 (nibble 0x4) -> 8 bytes of 0x44
        # w row 1: sixteen FP4 values = -1.0 (nibble 0xA) -> 8 bytes of 0xAA
        w_q = torch.tensor(
            [[0x44] * 8, [0xAA] * 8],
            dtype=torch.uint8,
            device=device,
        )
        # All block scales = E4M3 1.0 = 0x38. One block per row (K=16).
        x_scales = torch.full((M, K // BLOCK_SIZE), 0x38, dtype=torch.uint8, device=device)
        w_scales = torch.full((N, K // BLOCK_SIZE), 0x38, dtype=torch.uint8, device=device)

        y = torch.empty(M, N, device=device, dtype=torch.float16)
        return {
            "x_q": x_q,
            "x_scales": x_scales,
            "w_q": w_q,
            "w_scales": w_scales,
            "alpha": 1.0,
            "y": y,
            "M": M,
            "N": N,
            "K": K,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge cases with the minimum K = 16 (one block).
        tests.append(self._make_test_case(1, 2, 16, zero_x=True))
        tests.append(self._make_test_case(2, 4, 16))
        tests.append(self._make_test_case(3, 5, 32))

        # Power-of-2 shapes.
        tests.append(self._make_test_case(16, 16, 32))
        tests.append(self._make_test_case(32, 64, 64))
        tests.append(self._make_test_case(128, 128, 256))

        # Non-power-of-2 leading dims with valid K.
        tests.append(self._make_test_case(30, 50, 64))
        tests.append(self._make_test_case(100, 200, 128))
        tests.append(self._make_test_case(255, 100, 128, alpha=0.125))

        # Realistic attention-projection shape.
        tests.append(self._make_test_case(512, 1024, 1024, alpha=1.0 / 64.0))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # Verbatim from AutoKernel Table 5: the row where Triton hit 2,898 TF/s
        # against CUTLASS's 1,777 TF/s (1.63x speedup). A correct FP4 tensor
        # core submission at this shape directly validates the paper's claim.
        return self._make_test_case(2048, 18432, 3072, alpha=1.0 / 64.0)
