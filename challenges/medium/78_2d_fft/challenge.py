import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="2D FFT",
            atol=1e-02,
            rtol=1e-02,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(self, signal: torch.Tensor, spectrum: torch.Tensor, M: int, N: int):
        assert signal.shape == (M * N * 2,)
        assert spectrum.shape == (M * N * 2,)
        assert signal.dtype == torch.float32
        assert spectrum.dtype == torch.float32
        assert signal.device == spectrum.device

        sig_ri = signal.view(M, N, 2)
        sig_c = torch.complex(sig_ri[..., 0].contiguous(), sig_ri[..., 1].contiguous())
        spec_c = torch.fft.fft2(sig_c)
        spec_ri = torch.stack((spec_c.real, spec_c.imag), dim=-1).contiguous()
        spectrum.copy_(spec_ri.view(-1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "signal": (ctypes.POINTER(ctypes.c_float), "in"),
            "spectrum": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N = 2, 2
        signal = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device="cuda", dtype=dtype)
        spectrum = torch.empty(M * N * 2, device="cuda", dtype=dtype)
        return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        cases = []

        def make_case(M, N, low=-1.0, high=1.0):
            signal = torch.empty(M * N * 2, device="cuda", dtype=dtype).uniform_(low, high)
            spectrum = torch.empty(M * N * 2, device="cuda", dtype=dtype)
            return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}

        def make_zero_case(M, N):
            signal = torch.zeros(M * N * 2, device="cuda", dtype=dtype)
            spectrum = torch.empty(M * N * 2, device="cuda", dtype=dtype)
            return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}

        def make_impulse_case(M, N):
            signal = torch.zeros(M * N * 2, device="cuda", dtype=dtype)
            signal[0] = 1.0
            spectrum = torch.empty(M * N * 2, device="cuda", dtype=dtype)
            return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}

        # Edge cases: small sizes
        cases.append(make_impulse_case(1, 1))
        cases.append(make_zero_case(2, 2))
        cases.append(make_case(1, 4))

        # Power-of-2 sizes
        cases.append(make_case(16, 16))
        cases.append(make_case(32, 64))

        # Non-power-of-2 sizes
        cases.append(make_case(3, 5))
        cases.append(make_case(30, 30))

        # Mixed positive/negative values
        cases.append(make_case(100, 200, low=-5.0, high=5.0))

        # Realistic sizes
        cases.append(make_case(256, 256))
        cases.append(make_case(512, 512))

        return cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N = 2048, 2048
        signal = torch.empty(M * N * 2, device="cuda", dtype=dtype).normal_(0.0, 1.0)
        spectrum = torch.empty(M * N * 2, device="cuda", dtype=dtype)
        return {"signal": signal, "spectrum": spectrum, "M": M, "N": N}
