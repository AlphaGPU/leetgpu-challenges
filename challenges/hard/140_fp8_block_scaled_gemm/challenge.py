import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

# DeepSeek-V3 / DeepGEMM style: 1x128 block scales along the reduction dimension.
BLOCK_SIZE = 128


class Challenge(ChallengeBase):
    name = "FP8 Block-Scaled GEMM"
    # Dequantized FP8 GEMM into float16. Reference accumulates in float64; a
    # float32 accumulation over K=16,384 differs by ~1e-2 in FP16 space, while
    # skipping a scale, using the wrong block size or treating E4M3 as uint8
    # changes outputs by O(1) or more. atol/rtol of 0.1 / 0.05 match NVFP4.
    atol = 1e-01
    rtol = 5e-02
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x_q: torch.Tensor,
        x_scales: torch.Tensor,
        w_q: torch.Tensor,
        w_scales: torch.Tensor,
        y: torch.Tensor,
        M: int,
        N: int,
        K: int,
    ):
        assert K % BLOCK_SIZE == 0, "K must be divisible by 128"
        assert x_q.shape == (M, K)
        assert w_q.shape == (N, K)
        assert x_scales.shape == (M, K // BLOCK_SIZE)
        assert w_scales.shape == (N, K // BLOCK_SIZE)
        assert y.shape == (M, N)
        assert x_q.dtype == w_q.dtype == torch.uint8
        assert x_scales.dtype == w_scales.dtype == torch.float32
        assert y.dtype == torch.float16

        # Decode E4M3 payloads and apply 1x128 block scales in float64.
        n_blocks = K // BLOCK_SIZE
        x = x_q.view(torch.float8_e4m3fn).to(torch.float64)
        w = w_q.view(torch.float8_e4m3fn).to(torch.float64)
        x = (x.reshape(M, n_blocks, BLOCK_SIZE) * x_scales.to(torch.float64).unsqueeze(-1)).reshape(
            M, K
        )
        w = (w.reshape(N, n_blocks, BLOCK_SIZE) * w_scales.to(torch.float64).unsqueeze(-1)).reshape(
            N, K
        )
        y.copy_((x @ w.T).to(torch.float16))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x_q": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "x_scales": (ctypes.POINTER(ctypes.c_float), "in"),
            "w_q": (ctypes.POINTER(ctypes.c_uint8), "in"),
            "w_scales": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_uint16), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def _make(
        self,
        M: int,
        N: int,
        K: int,
        seed: int = 0,
        zero_x: bool = False,
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        assert K % BLOCK_SIZE == 0
        torch.manual_seed(seed)
        device = self.device
        if zero_x:
            x_q = torch.zeros(M, K, dtype=torch.uint8, device=device)
        else:
            # Sample finite E4M3 codes: avoid NaN (0x7F/0xFF) and keep magnitudes modest.
            x_f = torch.empty(M, K, device=device).uniform_(-2.0, 2.0)
            x_q = x_f.to(torch.float8_e4m3fn).view(torch.uint8)
        w_f = torch.empty(N, K, device=device).uniform_(-2.0, 2.0)
        w_q = w_f.to(torch.float8_e4m3fn).view(torch.uint8)
        x_scales = (torch.empty(M, K // BLOCK_SIZE, device=device).uniform_(0.25, 1.5) * scale).to(
            torch.float32
        )
        w_scales = (torch.empty(N, K // BLOCK_SIZE, device=device).uniform_(0.25, 1.5) * scale).to(
            torch.float32
        )
        y = torch.empty(M, N, device=device, dtype=torch.float16)
        return {
            "x_q": x_q,
            "x_scales": x_scales,
            "w_q": w_q,
            "w_scales": w_scales,
            "y": y,
            "M": M,
            "N": N,
            "K": K,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        M, N, K = 2, 2, 128
        # E4M3: 1.0=0x38, 0.5=0x30, 2.0=0x40, -1.0=0xB8
        x_q = torch.empty(M, K, dtype=torch.uint8, device=device)
        x_q[0] = 0x38
        x_q[1] = 0x30
        w_q = torch.empty(N, K, dtype=torch.uint8, device=device)
        w_q[0] = 0x40
        w_q[1] = 0xB8
        x_scales = torch.ones(M, K // BLOCK_SIZE, dtype=torch.float32, device=device)
        w_scales = torch.ones(N, K // BLOCK_SIZE, dtype=torch.float32, device=device)
        y = torch.empty(M, N, device=device, dtype=torch.float16)
        return {
            "x_q": x_q,
            "x_scales": x_scales,
            "w_q": w_q,
            "w_scales": w_scales,
            "y": y,
            "M": M,
            "N": N,
            "K": K,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]
        # Zero activations -> zero output.
        tests.append(self._make(1, 4, 128, seed=1, zero_x=True))
        # Minimum K, non-power-of-two leading dims.
        tests.append(self._make(3, 5, 128, seed=2))
        # Several blocks along K.
        tests.append(self._make(8, 16, 256, seed=3))
        tests.append(self._make(16, 32, 512, seed=4))
        # Unit scales forced via scale~0 then clamp — covered by random scales.
        tests.append(self._make(64, 64, 512, seed=5))
        # Non-power-of-two M/N with larger K.
        tests.append(self._make(30, 50, 1024, seed=6))
        tests.append(self._make(127, 63, 1024, seed=7))
        # Tiny scales: catches forgetting to multiply scales.
        t = self._make(32, 32, 256, seed=8, scale=0.01)
        tests.append(t)
        # Realistic projection shape (batch tokens x hidden).
        tests.append(self._make(512, 1024, 2048, seed=9))
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Decode-side projection: many tokens, large hidden, K divisible by 128.
        return self._make(2048, 4096, 4096, seed=0)
