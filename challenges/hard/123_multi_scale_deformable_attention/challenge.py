import ctypes
from typing import Any, Dict, List, Sequence, Tuple

import torch
from core.challenge_base import ChallengeBase

# Bilinear corners as (row offset, column offset).
_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))


class Challenge(ChallengeBase):
    name = "Multi-Scale Deformable Attention"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        sampling_loc: torch.Tensor,
        attn_weight: torch.Tensor,
        output: torch.Tensor,
        num_queries: int,
        num_heads: int,
        head_dim: int,
        num_levels: int,
        num_points: int,
    ):
        assert value.ndim == 3
        assert value.shape[1:] == (num_heads, head_dim)
        assert spatial_shapes.shape == (num_levels, 2)
        assert sampling_loc.shape == (num_queries, num_heads, num_levels, num_points, 2)
        assert attn_weight.shape == (num_queries, num_heads, num_levels, num_points)
        assert output.shape == (num_queries, num_heads, head_dim)
        assert value.dtype == torch.float32
        assert sampling_loc.dtype == torch.float32
        assert attn_weight.dtype == torch.float32
        assert output.dtype == torch.float32
        assert spatial_shapes.dtype == torch.int32

        shapes = spatial_shapes.to(torch.int64)
        level_h = shapes[:, 0].view(1, 1, num_levels, 1)
        level_w = shapes[:, 1].view(1, 1, num_levels, 1)
        sizes = shapes[:, 0] * shapes[:, 1]
        level_start = (torch.cumsum(sizes, dim=0) - sizes).view(1, 1, num_levels, 1)

        # Un-normalize the sampling locations into the pixel grid of their own level.
        x = sampling_loc[..., 0] * level_w.to(value.dtype) - 0.5
        y = sampling_loc[..., 1] * level_h.to(value.dtype) - 0.5
        x_floor = torch.floor(x)
        y_floor = torch.floor(y)
        dx = x - x_floor
        dy = y - y_floor
        x0 = x_floor.to(torch.int64)
        y0 = y_floor.to(torch.int64)

        # Head-major layout so one gather per bilinear corner serves every query.
        value_t = value.permute(1, 0, 2).contiguous()
        acc = torch.zeros(num_heads, num_queries, head_dim, device=value.device, dtype=value.dtype)

        for row_offset, col_offset in _CORNERS:
            xi = x0 + col_offset
            yi = y0 + row_offset
            inside = (xi >= 0) & (xi < level_w) & (yi >= 0) & (yi < level_h)

            # Out-of-range corners contribute zero, so they can safely read index 0.
            index = level_start + yi * level_w + xi
            index = torch.where(inside, index, torch.zeros_like(index))
            weight = (dx if col_offset == 1 else 1.0 - dx) * (dy if row_offset == 1 else 1.0 - dy)
            weight = torch.where(inside, weight * attn_weight, torch.zeros_like(weight))

            flat_index = index.permute(1, 0, 2, 3).reshape(num_heads, -1, 1)
            gathered = torch.gather(value_t, 1, flat_index.expand(-1, -1, head_dim))
            gathered = gathered.view(num_heads, num_queries, num_levels, num_points, head_dim)
            acc += (gathered * weight.permute(1, 0, 2, 3).unsqueeze(-1)).sum(dim=(2, 3))

        output.copy_(acc.permute(1, 0, 2))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "value": (ctypes.POINTER(ctypes.c_float), "in"),
            "spatial_shapes": (ctypes.POINTER(ctypes.c_int), "in"),
            "sampling_loc": (ctypes.POINTER(ctypes.c_float), "in"),
            "attn_weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "num_queries": (ctypes.c_int, "in"),
            "num_heads": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
            "num_levels": (ctypes.c_int, "in"),
            "num_points": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self,
        num_queries: int,
        num_heads: int,
        head_dim: int,
        shapes: Sequence[Tuple[int, int]],
        num_points: int,
        loc_low: float = 0.0,
        loc_high: float = 1.0,
        softmax_weights: bool = True,
        zero_value: bool = False,
        seed: int = 0,
    ) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        torch.manual_seed(seed)

        num_levels = len(shapes)
        total_spatial = sum(h * w for h, w in shapes)
        spatial_shapes = torch.tensor(shapes, device=device, dtype=torch.int32)

        if zero_value:
            value = torch.zeros(total_spatial, num_heads, head_dim, device=device, dtype=dtype)
        else:
            value = torch.randn(total_spatial, num_heads, head_dim, device=device, dtype=dtype)

        shape = (num_queries, num_heads, num_levels, num_points)
        sampling_loc = torch.rand(*shape, 2, device=device, dtype=dtype)
        sampling_loc = sampling_loc * (loc_high - loc_low) + loc_low

        raw = torch.randn(*shape, device=device, dtype=dtype)
        if softmax_weights:
            # Real detectors feed a softmax over the num_levels * num_points samples.
            attn_weight = torch.softmax(raw.view(num_queries, num_heads, -1), dim=-1).view(shape)
        else:
            attn_weight = raw

        output = torch.empty(num_queries, num_heads, head_dim, device=device, dtype=dtype)
        return {
            "value": value,
            "spatial_shapes": spatial_shapes,
            "sampling_loc": sampling_loc,
            "attn_weight": attn_weight,
            "output": output,
            "num_queries": num_queries,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_levels": num_levels,
            "num_points": num_points,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32

        # Two levels: a 2x2 feature map followed by a 1x1 feature map.
        spatial_shapes = torch.tensor([[2, 2], [1, 1]], device=device, dtype=torch.int32)
        value = torch.tensor(
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[10.0]]], device=device, dtype=dtype
        )
        sampling_loc = torch.tensor(
            [
                [[[[0.5, 0.5]], [[0.5, 0.5]]]],
                [[[[1.0, 1.0]], [[0.5, 0.5]]]],
            ],
            device=device,
            dtype=dtype,
        )
        attn_weight = torch.tensor(
            [
                [[[0.5], [0.5]]],
                [[[0.75], [0.25]]],
            ],
            device=device,
            dtype=dtype,
        )
        output = torch.empty(2, 1, 1, device=device, dtype=dtype)
        return {
            "value": value,
            "spatial_shapes": spatial_shapes,
            "sampling_loc": sampling_loc,
            "attn_weight": attn_weight,
            "output": output,
            "num_queries": 2,
            "num_heads": 1,
            "head_dim": 1,
            "num_levels": 2,
            "num_points": 1,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # Edge case: a single query sampling a single point of a single 1x1 level
        tests.append(self._make_test_case(1, 1, 1, [(1, 1)], 1, seed=0))

        # Edge case: one level, a 2x3 map, locations that fall on and past the border
        tests.append(self._make_test_case(2, 1, 2, [(2, 3)], 2, 0.0, 1.0, seed=1))

        # Edge case: a 1-pixel coarse level, so three of four corners are out of range
        tests.append(self._make_test_case(3, 2, 4, [(2, 2), (1, 1)], 1, seed=2))

        # Zero value tensor: every sample interpolates to zero
        tests.append(self._make_test_case(4, 2, 8, [(4, 4), (2, 2)], 2, zero_value=True, seed=3))

        # Locations outside [0, 1] with raw (non-normalized) negative weights
        tests.append(
            self._make_test_case(
                128, 4, 32, [(16, 16), (8, 8)], 4, -0.1, 1.1, softmax_weights=False, seed=5
            )
        )

        # Power-of-2 shapes, three levels
        tests.append(self._make_test_case(256, 8, 32, [(32, 32), (16, 16), (8, 8)], 4, seed=6))
        tests.append(self._make_test_case(1024, 4, 64, [(16, 16), (8, 8)], 8, seed=7))

        # Non-power-of-2 shapes with non-square, tiny levels
        tests.append(self._make_test_case(100, 3, 30, [(15, 7), (5, 3)], 3, -0.2, 1.2, seed=8))
        tests.append(
            self._make_test_case(255, 5, 17, [(13, 11), (7, 5), (3, 3), (2, 1)], 5, seed=9)
        )

        # Realistic Deformable DETR encoder step over a four-level pyramid
        tests.append(
            self._make_test_case(5440, 8, 32, [(64, 64), (32, 32), (16, 16), (8, 8)], 4, seed=10)
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # Encoder self-attention: every pixel of the pyramid is a query.
        shapes = [(128, 128), (64, 64), (32, 32), (16, 16)]
        num_queries = sum(h * w for h, w in shapes)
        return self._make_test_case(num_queries, 8, 32, shapes, 4, seed=42)
