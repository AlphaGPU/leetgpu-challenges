import ctypes
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase

# RMSNorm epsilon, matching Llama-style blocks in this repo.
EPS = 1e-5


class Challenge(ChallengeBase):
    name = "Persistent Multi-Layer MLP"
    # Each layer is RMSNorm, three GEMMs, SiLU, and a residual add. With the 0.02 weight scale
    # used here, a float64 recomputation of the two-layer example and of an 8-layer mid-size
    # case differs from float32 by well under 1e-5. atol = rtol = 1e-4 absorbs other float32
    # reduction orders without hiding a skipped layer, a post-norm residual, or a missing SiLU.
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        rms_w: torch.Tensor,
        W_gate: torch.Tensor,
        W_up: torch.Tensor,
        W_down: torch.Tensor,
        output: torch.Tensor,
        M: int,
        d_model: int,
        d_ffn: int,
        L: int,
    ):
        assert x.shape == (M, d_model)
        assert rms_w.shape == (L, d_model)
        assert W_gate.shape == (L, d_model, d_ffn)
        assert W_up.shape == (L, d_model, d_ffn)
        assert W_down.shape == (L, d_ffn, d_model)
        assert output.shape == (M, d_model)
        assert (
            x.dtype
            == rms_w.dtype
            == W_gate.dtype
            == W_up.dtype
            == W_down.dtype
            == output.dtype
            == torch.float32
        )

        h = x
        for layer in range(L):
            normed = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * rms_w[layer]
            gate = F.silu(normed @ W_gate[layer])
            up = normed @ W_up[layer]
            h = h + (gate * up) @ W_down[layer]
        output.copy_(h)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "rms_w": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_gate": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_up": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_down": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "d_ffn": (ctypes.c_int, "in"),
            "L": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        M: int,
        d_model: int,
        d_ffn: int,
        L: int,
        seed: int,
        zero_x: bool = False,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        device = self.device
        dtype = torch.float32
        if zero_x:
            x = torch.zeros(M, d_model, device=device, dtype=dtype)
        else:
            x = torch.randn(M, d_model, device=device, dtype=dtype) * 0.1
        rms_w = torch.empty(L, d_model, device=device, dtype=dtype).uniform_(0.8, 1.2)
        W_gate = torch.randn(L, d_model, d_ffn, device=device, dtype=dtype) * 0.02
        W_up = torch.randn(L, d_model, d_ffn, device=device, dtype=dtype) * 0.02
        W_down = torch.randn(L, d_ffn, d_model, device=device, dtype=dtype) * 0.02
        output = torch.empty(M, d_model, device=device, dtype=dtype)
        return {
            "x": x,
            "rms_w": rms_w,
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "output": output,
            "M": M,
            "d_model": d_model,
            "d_ffn": d_ffn,
            "L": L,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        M, d_model, d_ffn, L = 2, 2, 2, 2
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)
        rms_w = torch.ones(L, d_model, device=device, dtype=dtype)
        W_gate = torch.zeros(L, d_model, d_ffn, device=device, dtype=dtype)
        W_up = torch.zeros(L, d_model, d_ffn, device=device, dtype=dtype)
        W_down = torch.zeros(L, d_ffn, d_model, device=device, dtype=dtype)
        # Layer 0 is a rank-1 SwiGLU on the first coordinate; layer 1 is the same on the second.
        W_gate[0, 0, 0] = 1.0
        W_up[0, 0, 0] = 1.0
        W_down[0, 0, 0] = 1.0
        W_gate[1, 1, 1] = 1.0
        W_up[1, 1, 1] = 1.0
        W_down[1, 1, 1] = 1.0
        output = torch.empty(M, d_model, device=device, dtype=dtype)
        return {
            "x": x,
            "rms_w": rms_w,
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "output": output,
            "M": M,
            "d_model": d_model,
            "d_ffn": d_ffn,
            "L": L,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Independently computed two-layer example. Catches applying both layers' weights at
        # once (no residual recurrence), swapping pre-norm for post-norm, or skipping SiLU.
        example = self.generate_example_test()
        expected_output = torch.tensor(
            [[2.6088225841522217, 0.0], [0.0, 2.6088225841522217]],
            device=self.device,
            dtype=torch.float32,
        )
        out = torch.empty_like(example["output"])
        self.reference_impl(
            example["x"],
            example["rms_w"],
            example["W_gate"],
            example["W_up"],
            example["W_down"],
            out,
            example["M"],
            example["d_model"],
            example["d_ffn"],
            example["L"],
        )
        assert torch.allclose(out, expected_output, atol=1e-5, rtol=1e-5), out
        tests.append(example)

        # Single token, single layer: reduces to pre-norm residual SwiGLU. Distinct from the
        # existing single-layer SwiGLU MLP (no RMSNorm, no residual).
        tests.append(self._make_test_case(1, 4, 8, 1, seed=0))

        # Two rows, two layers.
        tests.append(self._make_test_case(2, 4, 8, 2, seed=1))

        # Zero input: every residual stays 0, so the output must be all zeros after any L.
        tests.append(self._make_test_case(4, 8, 16, 4, seed=2, zero_x=True))

        # Power-of-2 sizes, four layers.
        tests.append(self._make_test_case(16, 32, 64, 4, seed=3))

        # Power-of-2, eight stacked layers (the persistence / grid-sync path).
        tests.append(self._make_test_case(64, 64, 128, 8, seed=4))

        # Non-power-of-2 M and widths, three layers.
        tests.append(self._make_test_case(30, 24, 40, 3, seed=5))

        # Non-power-of-2 everything, including L=5.
        tests.append(self._make_test_case(100, 60, 84, 5, seed=6))

        # Realistic small prefill: 256 tokens, d_model=512, 4 layers.
        tests.append(self._make_test_case(256, 512, 1408, 4, seed=7))

        # Longer stack, mid-size model.
        tests.append(self._make_test_case(128, 1024, 2816, 8, seed=8))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Eight pre-norm SwiGLU layers, M=2048, d_model=1024, d_ffn=2816. Float32 working set
        # is ~293MB, so 5 copies fit comfortably in the T4's 16GB.
        return self._make_test_case(2048, 1024, 2816, 8, seed=42)
