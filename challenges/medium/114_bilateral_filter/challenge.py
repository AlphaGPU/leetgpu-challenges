import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Bilateral Filter"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        image: torch.Tensor,
        output: torch.Tensor,
        H: int,
        W: int,
        spatial_sigma: float,
        range_sigma: float,
        radius: int,
    ):
        assert image.shape == (H * W,)
        assert output.shape == (H * W,)
        assert image.dtype == torch.float32
        assert output.dtype == torch.float32

        r = int(radius)
        img = image.view(H, W)

        yi = torch.arange(H, device=image.device)
        xi = torch.arange(W, device=image.device)

        out = torch.zeros(H, W, device=image.device, dtype=torch.float32)
        norm = torch.zeros(H, W, device=image.device, dtype=torch.float32)
        inv_2ss2 = 1.0 / (2.0 * float(spatial_sigma) ** 2)
        inv_2rs2 = 1.0 / (2.0 * float(range_sigma) ** 2)

        for dy in range(-r, r + 1):
            rows = torch.clamp(yi + dy, 0, H - 1)
            for dx in range(-r, r + 1):
                cols = torch.clamp(xi + dx, 0, W - 1)
                neighbor = img.index_select(0, rows).index_select(1, cols)
                spatial_weight = math.exp(-(dy * dy + dx * dx) * inv_2ss2)
                range_weight = torch.exp(-((neighbor - img) ** 2) * inv_2rs2)
                weight = spatial_weight * range_weight
                out += weight * neighbor
                norm += weight

        output.copy_(out.view(-1) / norm.view(-1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "image": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "H": (ctypes.c_int, "in"),
            "W": (ctypes.c_int, "in"),
            "spatial_sigma": (ctypes.c_float, "in"),
            "range_sigma": (ctypes.c_float, "in"),
            "radius": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        H, W = 3, 3
        image = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype
        )
        output = torch.zeros(H * W, device=self.device, dtype=dtype)
        return {
            "image": image,
            "output": output,
            "H": H,
            "W": W,
            "spatial_sigma": 1.0,
            "range_sigma": 0.5,
            "radius": 1,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # single_pixel
        H, W = 1, 1
        tests.append(
            {
                "image": torch.tensor([0.5], device=self.device, dtype=dtype),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 1.0,
                "range_sigma": 0.5,
                "radius": 1,
            }
        )

        # two_by_two_zeros
        H, W = 2, 2
        tests.append(
            {
                "image": torch.zeros(H * W, device=self.device, dtype=dtype),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 1.0,
                "range_sigma": 0.5,
                "radius": 1,
            }
        )

        # three_by_three_ring (matches example)
        H, W = 3, 3
        tests.append(
            {
                "image": torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype
                ),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 1.0,
                "range_sigma": 0.5,
                "radius": 1,
            }
        )

        # four_by_four_negatives
        H, W = 4, 4
        tests.append(
            {
                "image": torch.tensor(
                    [
                        -1.0,
                        -1.0,
                        1.0,
                        1.0,
                        -1.0,
                        -1.0,
                        1.0,
                        1.0,
                        -1.0,
                        -1.0,
                        1.0,
                        1.0,
                        -1.0,
                        -1.0,
                        1.0,
                        1.0,
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 1.5,
                "range_sigma": 0.8,
                "radius": 1,
            }
        )

        # power_of_two_16x16
        H, W = 16, 16
        tests.append(
            {
                "image": torch.empty(H * W, device=self.device, dtype=dtype).uniform_(0.0, 1.0),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 1.5,
                "range_sigma": 0.3,
                "radius": 2,
            }
        )

        # power_of_two_64x64_mixed
        H, W = 64, 64
        tests.append(
            {
                "image": torch.empty(H * W, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 2.0,
                "range_sigma": 0.5,
                "radius": 2,
            }
        )

        # non_power_of_two_100x100
        H, W = 100, 100
        tests.append(
            {
                "image": torch.empty(H * W, device=self.device, dtype=dtype).uniform_(0.0, 1.0),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 2.0,
                "range_sigma": 0.3,
                "radius": 3,
            }
        )

        # non_power_of_two_255x255_mixed
        H, W = 255, 255
        tests.append(
            {
                "image": torch.empty(H * W, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 1.5,
                "range_sigma": 0.4,
                "radius": 2,
            }
        )

        # realistic_512x512
        H, W = 512, 512
        tests.append(
            {
                "image": torch.empty(H * W, device=self.device, dtype=dtype).uniform_(0.0, 1.0),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 2.0,
                "range_sigma": 0.2,
                "radius": 3,
            }
        )

        # realistic_1000x1000
        H, W = 1000, 1000
        tests.append(
            {
                "image": torch.empty(H * W, device=self.device, dtype=dtype).uniform_(0.0, 1.0),
                "output": torch.zeros(H * W, device=self.device, dtype=dtype),
                "H": H,
                "W": W,
                "spatial_sigma": 3.0,
                "range_sigma": 0.1,
                "radius": 5,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        H, W = 2048, 2048
        return {
            "image": torch.empty(H * W, device=self.device, dtype=dtype).uniform_(0.0, 1.0),
            "output": torch.zeros(H * W, device=self.device, dtype=dtype),
            "H": H,
            "W": W,
            "spatial_sigma": 3.0,
            "range_sigma": 0.1,
            "radius": 5,
        }
