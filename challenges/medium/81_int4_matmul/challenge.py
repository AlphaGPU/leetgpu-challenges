import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="INT4 Weight-Only Quantized MatMul",
            atol=1e-02,
            rtol=1e-02,
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

        # Unpack INT4 weights from packed uint8 bytes.
        # w_q[n, i] stores two weights: w[n, 2*i] in the high nibble (bits 7:4)
        # and w[n, 2*i+1] in the low nibble (bits 3:0).
        # INT4 values are stored unsigned (0–15) with an offset of 8,
        # so the signed value is nibble - 8, giving range [-8, 7].
        w_high = ((w_q >> 4) & 0xF).to(torch.int32) - 8  # [N, K//2]
        w_low = (w_q & 0xF).to(torch.int32) - 8  # [N, K//2]

        # Interleave high and low nibbles to reconstruct [N, K]
        w_int = torch.stack([w_high, w_low], dim=-1).reshape(N, K)  # [N, K]

        # Apply group-wise scales: dequantize each group
        n_groups = K // group_size
        w_groups = w_int.reshape(N, n_groups, group_size).float()  # [N, n_groups, group_size]
        scales_f = scales.float().unsqueeze(-1)  # [N, n_groups, 1]
        w_dequant = (w_groups * scales_f).reshape(N, K)  # [N, K]

        # MatMul: x [M, K] @ w_dequant.T [K, N] = y [M, N]
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
            x = torch.randn(M, K, device=device, dtype=torch.float16)
        # Random packed INT4 weights: each byte holds two nibbles in [0,15]
        w_q = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
        # Small positive scales to keep magnitudes reasonable
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
        # Packed INT4 weights (high nibble first).
        # Row 0: weights [1,1,1,1]  → nibbles stored as [9,9,9,9] → bytes [0x99, 0x99] = [153, 153]
        # Row 1: weights [2,2,2,2]  → nibbles [10,10,10,10]      → bytes [0xAA, 0xAA] = [170, 170]
        # Row 2: weights [-1,-1,-1,-1] → nibbles [7,7,7,7]       → bytes [0x77, 0x77] = [119, 119]
        # Row 3: weights [0,0,0,0]  → nibbles [8,8,8,8]          → bytes [0x88, 0x88] = [136, 136]
        w_q = torch.tensor(
            [[153, 153], [170, 170], [119, 119], [136, 136]],
            dtype=torch.uint8,
            device=device,
        )
        # One scale per group (group_size=2 → 2 groups per row), all 0.5
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

        # Edge cases — tiny K, small group_size
        tests.append(self._make_test_case(1, 2, 4, 2, zero_x=True))
        tests.append(self._make_test_case(2, 4, 4, 2))
        tests.append(self._make_test_case(3, 5, 8, 4))

        # Power-of-2 sizes
        tests.append(self._make_test_case(16, 16, 32, 16))
        tests.append(self._make_test_case(32, 64, 64, 32))
        tests.append(self._make_test_case(64, 128, 128, 64))

        # Non-power-of-2 sizes
        tests.append(self._make_test_case(30, 50, 64, 32))
        tests.append(self._make_test_case(100, 200, 128, 64))
        tests.append(self._make_test_case(255, 100, 128, 64))

        # Realistic LLM inference sizes
        tests.append(self._make_test_case(128, 256, 512, 128))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # Typical LLM weight matrix: 4096×4096 with group_size=128
        return self._make_test_case(4096, 4096, 4096, 128)
