import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Integral Image"
    # Each output is a sum of up to H*W terms; parallel scans reorder the fp32 additions, so
    # rounding error grows with the partial-sum magnitude. 1e-2 absorbs that while the
    # integer-valued targeted tests below still expose any missing or extra term (error >= 1).
    atol = 1e-02
    rtol = 1e-02
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, H: int, W: int):
        assert input.shape == (H, W)
        assert output.shape == (H, W)
        assert input.dtype == torch.float32
        assert output.dtype == torch.float32

        result = torch.cumsum(torch.cumsum(input, dim=0), dim=1)
        output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "H": (ctypes.c_int, "in"),
            "W": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        input = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            device=self.device,
            dtype=dtype,
        )
        output = torch.empty((3, 3), device=self.device, dtype=dtype)
        return {"input": input, "output": output, "H": 3, "W": 3}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # single_element
        tests.append(
            {
                "input": torch.tensor([[7.0]], device=self.device, dtype=dtype),
                "output": torch.empty((1, 1), device=self.device, dtype=dtype),
                "H": 1,
                "W": 1,
            }
        )

        # single_row
        tests.append(
            {
                "input": torch.tensor([[1.0, -2.0, 3.0, -4.0]], device=self.device, dtype=dtype),
                "output": torch.empty((1, 4), device=self.device, dtype=dtype),
                "H": 1,
                "W": 4,
            }
        )

        # single_col
        tests.append(
            {
                "input": torch.tensor(
                    [[2.0], [5.0], [-1.0], [3.0]], device=self.device, dtype=dtype
                ),
                "output": torch.empty((4, 1), device=self.device, dtype=dtype),
                "H": 4,
                "W": 1,
            }
        )

        # all_zeros_16x16
        tests.append(
            {
                "input": torch.zeros((16, 16), device=self.device, dtype=dtype),
                "output": torch.empty((16, 16), device=self.device, dtype=dtype),
                "H": 16,
                "W": 16,
            }
        )

        # power_of_2_square_32x32
        tests.append(
            {
                "input": torch.empty((32, 32), device=self.device, dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty((32, 32), device=self.device, dtype=dtype),
                "H": 32,
                "W": 32,
            }
        )

        # power_of_2_square_128x128
        tests.append(
            {
                "input": torch.empty((128, 128), device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "output": torch.empty((128, 128), device=self.device, dtype=dtype),
                "H": 128,
                "W": 128,
            }
        )

        # non_power_of_2_30x30
        tests.append(
            {
                "input": torch.empty((30, 30), device=self.device, dtype=dtype).uniform_(-3.0, 3.0),
                "output": torch.empty((30, 30), device=self.device, dtype=dtype),
                "H": 30,
                "W": 30,
            }
        )

        # non_power_of_2_100x100_negative
        tests.append(
            {
                "input": torch.empty((100, 100), device=self.device, dtype=dtype).uniform_(
                    -10.0, 0.0
                ),
                "output": torch.empty((100, 100), device=self.device, dtype=dtype),
                "H": 100,
                "W": 100,
            }
        )

        # non_square_255x33
        tests.append(
            {
                "input": torch.empty((255, 33), device=self.device, dtype=dtype).uniform_(
                    -2.0, 2.0
                ),
                "output": torch.empty((255, 33), device=self.device, dtype=dtype),
                "H": 255,
                "W": 33,
            }
        )

        # all_ones_5x7: output[i][j] = (i + 1) * (j + 1). Checked against independently computed
        # values; catches row-only / column-only scans and exclusive (shifted) scans.
        ones_expected = torch.tensor(
            [[(i + 1) * (j + 1) for j in range(7)] for i in range(5)], dtype=dtype
        )
        ones_out = torch.empty((5, 7), dtype=dtype)
        self.reference_impl(torch.ones((5, 7), dtype=dtype), ones_out, 5, 7)
        assert torch.equal(ones_out, ones_expected)
        tests.append(
            {
                "input": torch.ones((5, 7), device=self.device, dtype=dtype),
                "output": torch.empty((5, 7), device=self.device, dtype=dtype),
                "H": 5,
                "W": 7,
            }
        )

        # wide_3x300_integers: W > H with W not a multiple of a block size; catches swapped H/W
        # indexing (non-square in the other direction from 255x33) and missed row-scan tails
        tests.append(
            {
                "input": torch.randint(-3, 4, (3, 300), device=self.device).to(dtype),
                "output": torch.empty((3, 300), device=self.device, dtype=dtype),
                "H": 3,
                "W": 300,
            }
        )

        # tile_tails_257x513_integers: one past power-of-2 in both dims; catches dropped
        # carries between tiles/blocks in either scan direction (integer values -> exact sums)
        tests.append(
            {
                "input": torch.randint(-2, 3, (257, 513), device=self.device).to(dtype),
                "output": torch.empty((257, 513), device=self.device, dtype=dtype),
                "H": 257,
                "W": 513,
            }
        )

        # mixed_extreme_64x64: values span the full documented range [-100, 100]
        tests.append(
            {
                "input": torch.empty((64, 64), device=self.device, dtype=dtype).uniform_(
                    -100.0, 100.0
                ),
                "output": torch.empty((64, 64), device=self.device, dtype=dtype),
                "H": 64,
                "W": 64,
            }
        )

        # realistic_1024x1024
        tests.append(
            {
                "input": torch.empty((1024, 1024), device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "output": torch.empty((1024, 1024), device=self.device, dtype=dtype),
                "H": 1024,
                "W": 1024,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        H = 8192
        W = 8192
        return {
            "input": torch.empty((H, W), device=self.device, dtype=dtype).uniform_(-0.01, 0.01),
            "output": torch.empty((H, W), device=self.device, dtype=dtype),
            "H": H,
            "W": W,
        }
