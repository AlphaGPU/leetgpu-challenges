import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Bilinear Image Scaling",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free",
        )

    def reference_impl(
        self,
        image: torch.Tensor,
        output: torch.Tensor,
        H: int,
        W: int,
        H_out: int,
        W_out: int,
    ):
        assert image.shape == (H, W), f"Expected image.shape=({H},{W}), got {image.shape}"
        assert output.shape == (
            H_out,
            W_out,
        ), f"Expected output.shape=({H_out},{W_out}), got {output.shape}"
        assert image.dtype == torch.float32
        assert output.dtype == torch.float32
        assert image.device.type == "cuda"

        img = image.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        result = torch.nn.functional.interpolate(
            img, size=(H_out, W_out), mode="bilinear", align_corners=True
        )
        output.copy_(result.squeeze(0).squeeze(0))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "image": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "H": (ctypes.c_int, "in"),
            "W": (ctypes.c_int, "in"),
            "H_out": (ctypes.c_int, "in"),
            "W_out": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        image = torch.tensor(
            [[1.0, 3.0], [7.0, 9.0]],
            device="cuda",
            dtype=dtype,
        )
        output = torch.empty((3, 3), device="cuda", dtype=dtype)
        return {
            "image": image,
            "output": output,
            "H": 2,
            "W": 2,
            "H_out": 3,
            "W_out": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Edge case: 1x1 -> 1x1 (single pixel, no interpolation)
        tests.append(
            {
                "image": torch.tensor([[5.0]], device="cuda", dtype=dtype),
                "output": torch.empty((1, 1), device="cuda", dtype=dtype),
                "H": 1,
                "W": 1,
                "H_out": 1,
                "W_out": 1,
            }
        )

        # Edge case: 2x2 -> 2x2 (identity, no scaling)
        tests.append(
            {
                "image": torch.tensor([[-1.0, 2.0], [3.0, -4.0]], device="cuda", dtype=dtype),
                "output": torch.empty((2, 2), device="cuda", dtype=dtype),
                "H": 2,
                "W": 2,
                "H_out": 2,
                "W_out": 2,
            }
        )

        # Edge case: 3x3 -> 5x5 (small upsampling, includes zeros)
        tests.append(
            {
                "image": torch.zeros((3, 3), device="cuda", dtype=dtype),
                "output": torch.empty((5, 5), device="cuda", dtype=dtype),
                "H": 3,
                "W": 3,
                "H_out": 5,
                "W_out": 5,
            }
        )

        # Edge case: 4x4 -> 4x12 (width-only scaling, 3x)
        tests.append(
            {
                "image": torch.empty((4, 4), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty((4, 12), device="cuda", dtype=dtype),
                "H": 4,
                "W": 4,
                "H_out": 4,
                "W_out": 12,
            }
        )

        # Power-of-2: 16x16 -> 32x32 (all zeros)
        tests.append(
            {
                "image": torch.zeros((16, 16), device="cuda", dtype=dtype),
                "output": torch.empty((32, 32), device="cuda", dtype=dtype),
                "H": 16,
                "W": 16,
                "H_out": 32,
                "W_out": 32,
            }
        )

        # Power-of-2: 64x64 -> 128x128 (negative values)
        tests.append(
            {
                "image": torch.empty((64, 64), device="cuda", dtype=dtype).uniform_(-10.0, 0.0),
                "output": torch.empty((128, 128), device="cuda", dtype=dtype),
                "H": 64,
                "W": 64,
                "H_out": 128,
                "W_out": 128,
            }
        )

        # Power-of-2: 256x256 -> 512x512 (mixed values)
        tests.append(
            {
                "image": torch.empty((256, 256), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty((512, 512), device="cuda", dtype=dtype),
                "H": 256,
                "W": 256,
                "H_out": 512,
                "W_out": 512,
            }
        )

        # Non-power-of-2: 30x40 -> 60x80
        tests.append(
            {
                "image": torch.empty((30, 40), device="cuda", dtype=dtype).uniform_(-3.0, 3.0),
                "output": torch.empty((60, 80), device="cuda", dtype=dtype),
                "H": 30,
                "W": 40,
                "H_out": 60,
                "W_out": 80,
            }
        )

        # Non-power-of-2: 100x150 -> 255x400 (non-integer scale factors)
        tests.append(
            {
                "image": torch.empty((100, 150), device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
                "output": torch.empty((255, 400), device="cuda", dtype=dtype),
                "H": 100,
                "W": 150,
                "H_out": 255,
                "W_out": 400,
            }
        )

        # Realistic: 1024x1024 -> 2048x2048
        tests.append(
            {
                "image": torch.empty((1024, 1024), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty((2048, 2048), device="cuda", dtype=dtype),
                "H": 1024,
                "W": 1024,
                "H_out": 2048,
                "W_out": 2048,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        H = 4096
        W = 4096
        H_out = 8192
        W_out = 8192
        return {
            "image": torch.empty((H, W), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "output": torch.empty((H_out, W_out), device="cuda", dtype=dtype),
            "H": H,
            "W": W,
            "H_out": H_out,
            "W_out": W_out,
        }
