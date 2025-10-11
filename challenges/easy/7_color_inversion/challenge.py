import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Color Inversion",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, image: torch.Tensor, width: int, height: int):
        assert image.shape == (height * width * 4,)
        assert image.dtype == torch.uint8

        # Reshape to (height, width, 4) for easier processing
        image_reshaped = image.view(height, width, 4)

        # Invert RGB channels (first 3 channels), keep alpha unchanged
        image_reshaped[:, :, :3] = 255 - image_reshaped[:, :, :3]

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "image": (ctypes.POINTER(ctypes.c_ubyte), "inout"),
            "width": (ctypes.c_int, "in"),
            "height": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        width, height = 1, 2
        image = torch.tensor([255, 0, 128, 255, 0, 255, 0, 255], device="cuda", dtype=torch.uint8)
        return {
            "image": image,
            "width": width,
            "height": height
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        return [
            {
                "image": torch.tensor([
                    [[255, 0, 0, 255], [0, 255, 0, 255]],
                    [[0, 0, 255, 255], [128, 128, 128, 255]]
                ], dtype=torch.uint8, device="cuda").flatten(),
                "width": 2,
                "height": 2
            },
            {
                "image": torch.tensor([
                    [[100, 50, 200, 255]]
                ], dtype=torch.uint8, device="cuda").flatten(),
                "width": 1,
                "height": 1
            },
            {
                "image": torch.zeros((3, 4, 4), dtype=torch.uint8, device="cuda").flatten(),
                "width": 4,
                "height": 3
            },
            {
                "image": torch.full((5, 3, 4), 255, dtype=torch.uint8, device="cuda").flatten(),
                "width": 3,
                "height": 5
            },
            {
                "image": torch.tensor([
                    [[10, 20, 30, 50], [40, 50, 60, 100]],
                    [[70, 80, 90, 150], [100, 110, 120, 200]]
                ], dtype=torch.uint8, device="cuda").flatten(),
                "width": 2,
                "height": 2
            },
            {
                "image": torch.randint(0, 256, (64 * 64 * 4,), dtype=torch.uint8, device="cuda"),
                "width": 64,
                "height": 64
            },
            {
                "image": torch.randint(0, 256, (32 * 64 * 4,), dtype=torch.uint8, device="cuda"),
                "width": 64,
                "height": 32
            }
        ]

    def generate_performance_test(self) -> Dict[str, Any]:
        width, height = 4096, 5120
        size = width * height * 4
        return {
            "image": torch.randint(0, 256, (size,), device="cuda", dtype=torch.uint8),
            "width": width,
            "height": height
        }
