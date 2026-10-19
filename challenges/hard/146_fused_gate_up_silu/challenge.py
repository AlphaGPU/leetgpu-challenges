import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused Gate-Up Projection with SiLU"
    # output = silu(x @ W_gate) * (x @ W_up) with a packed [H, 2I] weight. Float64
    # reference keeps float32 GEMM + silu error around 1e-5 on the large cases;
    # swapping gate/up halves, dropping silu, or reading W as [2I, H] is off by >> 1e-3.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        W_gate_up: torch.Tensor,
        output: torch.Tensor,
        M: int,
        H: int,
        I: int,  # noqa: E741 — intermediate width, matches Llama "I" notation
    ):
        assert x.shape == (M, H)
        assert W_gate_up.shape == (H, 2 * I)
        assert output.shape == (M, I)
        assert x.dtype == W_gate_up.dtype == output.dtype == torch.float32

        # Accumulate the GEMM in float64 so float32 / TF32 solutions are checked against a
        # stable reference; silu and the multiply stay in the same precision.
        proj = x.double() @ W_gate_up.double()  # [M, 2I]
        gate = proj[:, :I]
        up = proj[:, I:]
        output.copy_((gate * torch.sigmoid(gate) * up).float())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_gate_up": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "I": (ctypes.c_int, "in"),
        }

    def _make(self, M, H, I, seed=0, x_scale=1.0, w_scale=0.02, zero_x=False):  # noqa: E741
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        if zero_x:
            x = torch.zeros(M, H, device=dev, dtype=f32)
        else:
            x = torch.randn(M, H, device=dev, dtype=f32) * x_scale
        W = torch.randn(H, 2 * I, device=dev, dtype=f32) * w_scale
        return {
            "x": x,
            "W_gate_up": W,
            "output": torch.full((M, I), 123.0, device=dev, dtype=f32),
            "M": M,
            "H": H,
            "I": I,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        # M=2, H=2, I=2. Hand-checked: silu(1)=sigmoid(1)≈0.731059, silu(0)=0.
        # Row 0: gate=[1,0], up=[1,0] → [silu(1)*1, silu(0)*0] = [0.731059, 0]
        # Row 1: gate=[0,1], up=[0,1] → [0, silu(1)*1] = [0, 0.731059]
        dev, f32 = self.device, torch.float32
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=dev, dtype=f32)
        # Columns 0..I-1 = gate, I..2I-1 = up. Identity on both halves.
        W = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            device=dev,
            dtype=f32,
        )
        return {
            "x": x,
            "W_gate_up": W,
            "output": torch.full((2, 2), 123.0, device=dev, dtype=f32),
            "M": 2,
            "H": 2,
            "I": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []
        tests.append(self.generate_example_test())

        # Smallest non-trivial case with asymmetric halves (catches gate/up swap).
        tests.append(self._make(1, 2, 2, seed=1, x_scale=1.0, w_scale=1.0))

        # Zero tokens → all-zero output; catches leaving the pre-filled garbage in output.
        tests.append(self._make(4, 8, 4, seed=2, zero_x=True, w_scale=1.0))

        # Negative and mid-range gate values: silu(z)=z*σ(z) is neither ReLU nor σ alone.
        tests.append(self._make(8, 8, 8, seed=3, x_scale=3.0, w_scale=1.0))

        # Odd intermediate width (not a power of 2): catches tiling that assumes I%32==0.
        tests.append(self._make(7, 16, 17, seed=4, w_scale=0.1))

        # Large |logits| so σ saturates; missing silu (plain gate*up) is far off.
        tests.append(self._make(16, 32, 64, seed=5, x_scale=5.0, w_scale=0.5))

        # Explicit half-swap trap: gate columns are ~10x larger than up columns.
        t = self._make(4, 8, 6, seed=10, x_scale=1.0, w_scale=1.0)
        t["W_gate_up"][:, :6] *= 10.0
        tests.append(t)

        # Layout trap: I != H, so reading W as [2I, H] (vertical pack) fails hard.
        tests.append(self._make(32, 64, 48, seed=6, w_scale=0.05))

        # Modest rectangular Llama-style widths.
        tests.append(self._make(64, 256, 512, seed=7, w_scale=0.02))

        # Tall-and-skinny then short-and-wide: stress both GEMM shapes.
        tests.append(self._make(512, 64, 128, seed=8, w_scale=0.05))
        tests.append(self._make(8, 512, 256, seed=9, w_scale=0.02))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # One Llama-3-8B-style gate-up step: M=4096 tokens, H=4096, I=11008.
        # x + W + out ≈ 64 + 360 + 180 = 604 MB; 5x headroom fits a T4's 16 GB.
        # GEMM is ~2*4096*4096*11008 ≈ 370 GFLOP → ~45–70 ms FP32 on T4.
        return self._make(4096, 4096, 11008, seed=0, w_scale=0.02)
