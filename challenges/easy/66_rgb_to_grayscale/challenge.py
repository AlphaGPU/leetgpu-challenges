import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="RGB to Grayscale", atol=1e-05, rtol=1e-05, num_gpus=1, access_tier="free"
        )

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, width: int, height: int):
        assert input.shape == (height * width * 3,)
        assert output.shape == (height * width,)
        assert input.dtype == output.dtype == torch.float32
        assert input.device == output.device

        # Reshape input to (height, width, 3) for easier processing
        rgb_image = input.view(height, width, 3)

        # Apply RGB to grayscale conversion: gray = 0.299*R + 0.587*G + 0.114*B
        grayscale = (
            0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]
        )

        # Flatten and store in output
        output.copy_(grayscale.flatten())

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "width": (ctypes.c_int, "in"),
            "height": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        width, height = 2, 2
        # RGB values for a 2x2 image
        # Pixel (0,0): R=255, G=0, B=0 (red)
        # Pixel (0,1): R=0, G=255, B=0 (green)
        # Pixel (1,0): R=0, G=0, B=255 (blue)
        # Pixel (1,1): R=128, G=128, B=128 (gray)
        input_data = torch.tensor(
            [
                255.0,
                0.0,
                0.0,  # red
                0.0,
                255.0,
                0.0,  # green
                0.0,
                0.0,
                255.0,  # blue
                128.0,
                128.0,
                128.0,  # gray
            ],
            device="cuda",
            dtype=torch.float32,
        )
        output = torch.zeros(width * height, device="cuda", dtype=torch.float32)
        return {
            "input": input_data,
            "output": output,
            "width": width,
            "height": height,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        test_cases = []

        # Small test cases
        test_cases.append(
            {
                "input": torch.tensor(
                    [255.0, 0.0, 0.0], device="cuda", dtype=torch.float32
                ),  # red pixel
                "output": torch.zeros(1, device="cuda", dtype=torch.float32),
                "width": 1,
                "height": 1,
            }
        )

        test_cases.append(
            {
                "input": torch.tensor(
                    [0.0, 255.0, 0.0], device="cuda", dtype=torch.float32
                ),  # green pixel
                "output": torch.zeros(1, device="cuda", dtype=torch.float32),
                "width": 1,
                "height": 1,
            }
        )

        test_cases.append(
            {
                "input": torch.tensor(
                    [0.0, 0.0, 255.0], device="cuda", dtype=torch.float32
                ),  # blue pixel
                "output": torch.zeros(1, device="cuda", dtype=torch.float32),
                "width": 1,
                "height": 1,
            }
        )

        # 2x2 test case
        test_cases.append(
            {
                "input": torch.tensor(
                    [
                        100.0,
                        150.0,
                        200.0,  # mixed color 1
                        50.0,
                        75.0,
                        100.0,  # mixed color 2
                        200.0,
                        100.0,
                        50.0,  # mixed color 3
                        75.0,
                        125.0,
                        175.0,  # mixed color 4
                    ],
                    device="cuda",
                    dtype=torch.float32,
                ),
                "output": torch.zeros(4, device="cuda", dtype=torch.float32),
                "width": 2,
                "height": 2,
            }
        )

        # Edge cases: zeros and max values
        test_cases.append(
            {
                "input": torch.zeros(3, device="cuda", dtype=torch.float32),
                "output": torch.zeros(1, device="cuda", dtype=torch.float32),
                "width": 1,
                "height": 1,
            }
        )

        test_cases.append(
            {
                "input": torch.full((3,), 255.0, device="cuda", dtype=torch.float32),
                "output": torch.zeros(1, device="cuda", dtype=torch.float32),
                "width": 1,
                "height": 1,
            }
        )

        # Larger test cases
        for size in [4, 8, 16, 32]:
            input_size = size * size * 3
            test_cases.append(
                {
                    "input": torch.randint(
                        0, 256, (input_size,), device="cuda", dtype=torch.float32
                    ),
                    "output": torch.zeros(size * size, device="cuda", dtype=torch.float32),
                    "width": size,
                    "height": size,
                }
            )

        # Larger realistic sizes
        for w, h in [(100, 100), (64, 48)]:
            test_cases.append(
                {
                    "input": torch.empty(h * w * 3, device="cuda", dtype=torch.float32).uniform_(
                        0.0, 255.0
                    ),
                    "output": torch.zeros(h * w, device="cuda", dtype=torch.float32),
                    "width": w,
                    "height": h,
                }
            )

        # Non-square images
        test_cases.append(
            {
                "input": torch.randint(
                    0, 256, (2 * 3 * 3,), device="cuda", dtype=torch.float32
                ),  # 2x3 image
                "output": torch.zeros(2 * 3, device="cuda", dtype=torch.float32),
                "width": 3,
                "height": 2,
            }
        )

        test_cases.append(
            {
                "input": torch.randint(
                    0, 256, (3 * 2 * 3,), device="cuda", dtype=torch.float32
                ),  # 3x2 image
                "output": torch.zeros(3 * 2, device="cuda", dtype=torch.float32),
                "width": 2,
                "height": 3,
            }
        )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        width, height = 2048, 2048
        input_size = width * height * 3
        output_size = width * height
        return {
            "input": torch.randint(0, 256, (input_size,), device="cuda", dtype=torch.float32),
            "output": torch.zeros(output_size, device="cuda", dtype=torch.float32),
            "width": width,
            "height": height,
        }
