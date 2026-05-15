import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Adaptive Layer Normalization",
            atol=1e-04,
            rtol=1e-04,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        X: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        output: torch.Tensor,
        B: int,
        N: int,
        D: int,
    ):
        assert X.shape == output.shape == (B, N, D)
        assert scale.shape == shift.shape == (B, D)
        assert X.dtype == scale.dtype == shift.dtype == output.dtype
        assert (
            X.device.type == "cuda"
            and scale.device.type == "cuda"
            and shift.device.type == "cuda"
            and output.device.type == "cuda"
        )

        eps = 1e-5
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (X - mean) / torch.sqrt(var + eps)
        output.copy_(x_norm * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "scale": (ctypes.POINTER(ctypes.c_float), "in"),
            "shift": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        B, N, D = 1, 2, 4
        X = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]],
            device="cuda",
            dtype=dtype,
        )
        scale = torch.tensor([[0.0, 1.0, 0.0, -0.5]], device="cuda", dtype=dtype)
        shift = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device="cuda", dtype=dtype)
        output = torch.empty((B, N, D), device="cuda", dtype=dtype)
        return {
            "X": X,
            "scale": scale,
            "shift": shift,
            "output": output,
            "B": B,
            "N": N,
            "D": D,
        }

    def _make_test(self, B: int, N: int, D: int, fill: str = "uniform") -> Dict[str, Any]:
        dtype = torch.float32
        if fill == "zeros":
            X = torch.zeros((B, N, D), device="cuda", dtype=dtype)
        elif fill == "negative":
            X = torch.empty((B, N, D), device="cuda", dtype=dtype).uniform_(-5.0, -0.1)
        elif fill == "mixed":
            X = torch.empty((B, N, D), device="cuda", dtype=dtype).uniform_(-3.0, 3.0)
        else:
            X = torch.empty((B, N, D), device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        scale = torch.empty((B, D), device="cuda", dtype=dtype).uniform_(-0.5, 0.5)
        shift = torch.empty((B, D), device="cuda", dtype=dtype).uniform_(-0.5, 0.5)
        output = torch.empty((B, N, D), device="cuda", dtype=dtype)
        return {
            "X": X,
            "scale": scale,
            "shift": shift,
            "output": output,
            "B": B,
            "N": N,
            "D": D,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(0)
        tests = []

        # basic example
        tests.append(self.generate_example_test())

        # tiny edge: single batch, single token, very small D
        tests.append(self._make_test(1, 1, 4))

        # tiny edge: D = 2
        tests.append(self._make_test(2, 3, 2))

        # zero input — output should equal shift exactly
        tests.append(self._make_test(2, 4, 8, fill="zeros"))

        # all-negative input
        tests.append(self._make_test(3, 5, 16, fill="negative"))

        # mixed values, power-of-2 D
        tests.append(self._make_test(2, 8, 64, fill="mixed"))

        # power-of-2 sizes
        tests.append(self._make_test(4, 16, 128))

        # non-power-of-2 sizes
        tests.append(self._make_test(3, 30, 100))

        # larger non-power-of-2
        tests.append(self._make_test(2, 255, 384))

        # realistic DiT-S/2 token count, modest D
        tests.append(self._make_test(4, 256, 384))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(1)
        # DiT-XL/2 inspired: B=16 batch, N=4096 patches (e.g. 64x64 latent grid), D=1152
        return self._make_test(16, 4096, 1152)
