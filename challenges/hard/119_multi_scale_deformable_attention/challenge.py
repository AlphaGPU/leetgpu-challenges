import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))


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
        assert value.dim() == 3
        assert value.shape[1:] == (num_heads, head_dim)
        assert spatial_shapes.shape == (num_levels, 2)
        assert sampling_loc.shape == (num_queries, num_heads, num_levels, num_points, 2)
        assert attn_weight.shape == (num_queries, num_heads, num_levels, num_points)
        assert output.shape == (num_queries, num_heads, head_dim)
        assert spatial_shapes.dtype == torch.int32
        assert value.dtype == sampling_loc.dtype == attn_weight.dtype == output.dtype

        dtype = value.dtype
        shapes = spatial_shapes.to(torch.int64)
        level_h = shapes[:, 0].reshape(1, 1, num_levels, 1)
        level_w = shapes[:, 1].reshape(1, 1, num_levels, 1)
        level_size = shapes[:, 0] * shapes[:, 1]
        level_start = (torch.cumsum(level_size, dim=0) - level_size).reshape(1, 1, num_levels, 1)

        # map normalized [0, 1] locations onto the pixel-center grid of each level
        x = sampling_loc[..., 0] * level_w.to(dtype) - 0.5
        y = sampling_loc[..., 1] * level_h.to(dtype) - 0.5
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        frac_x = x - x0
        frac_y = y - y0
        col0 = x0.to(torch.int64)
        row0 = y0.to(torch.int64)

        # value rows are (level-major) flattened pixels; fold the head axis into the row index
        value_flat = value.reshape(-1, head_dim)
        head_idx = torch.arange(num_heads, device=value.device, dtype=torch.int64).reshape(
            1, num_heads, 1, 1
        )

        num_samples = num_levels * num_points
        out = torch.zeros(num_queries, num_heads, head_dim, device=value.device, dtype=dtype)
        for row_off, col_off in CORNERS:
            row = row0 + row_off
            col = col0 + col_off
            inside = (row >= 0) & (row < level_h) & (col >= 0) & (col < level_w)
            weight_y = frac_y if row_off == 1 else 1.0 - frac_y
            weight_x = frac_x if col_off == 1 else 1.0 - frac_x
            weight = weight_y * weight_x * attn_weight * inside.to(dtype)

            row_c = torch.minimum(row.clamp(min=0), level_h - 1)
            col_c = torch.minimum(col.clamp(min=0), level_w - 1)
            flat = (level_start + row_c * level_w + col_c) * num_heads + head_idx
            gathered = value_flat.index_select(0, flat.reshape(-1))
            gathered = gathered.reshape(num_queries, num_heads, num_samples, head_dim)
            out = out + (gathered * weight.reshape(num_queries, num_heads, num_samples, 1)).sum(
                dim=2
            )

        output.copy_(out)

    def reference_impl_jax(
        self,
        value,
        spatial_shapes,
        sampling_loc,
        attn_weight,
        num_queries,
        num_heads,
        head_dim,
        num_levels,
        num_points,
    ):
        import jax.numpy as jnp

        dtype = value.dtype
        shapes = spatial_shapes.astype(jnp.int32)
        level_h = shapes[:, 0].reshape(1, 1, num_levels, 1)
        level_w = shapes[:, 1].reshape(1, 1, num_levels, 1)
        level_size = shapes[:, 0] * shapes[:, 1]
        level_start = (jnp.cumsum(level_size) - level_size).reshape(1, 1, num_levels, 1)

        x = sampling_loc[..., 0] * level_w.astype(dtype) - 0.5
        y = sampling_loc[..., 1] * level_h.astype(dtype) - 0.5
        x0 = jnp.floor(x)
        y0 = jnp.floor(y)
        frac_x = x - x0
        frac_y = y - y0
        col0 = x0.astype(jnp.int32)
        row0 = y0.astype(jnp.int32)

        value_flat = value.reshape(-1, head_dim)
        head_idx = jnp.arange(num_heads, dtype=jnp.int32).reshape(1, num_heads, 1, 1)

        num_samples = num_levels * num_points
        out = jnp.zeros((num_queries, num_heads, head_dim), dtype=dtype)
        for row_off, col_off in CORNERS:
            row = row0 + row_off
            col = col0 + col_off
            inside = (row >= 0) & (row < level_h) & (col >= 0) & (col < level_w)
            weight_y = frac_y if row_off == 1 else 1.0 - frac_y
            weight_x = frac_x if col_off == 1 else 1.0 - frac_x
            weight = weight_y * weight_x * attn_weight * inside.astype(dtype)

            row_c = jnp.minimum(jnp.maximum(row, 0), level_h - 1)
            col_c = jnp.minimum(jnp.maximum(col, 0), level_w - 1)
            flat = (level_start + row_c * level_w + col_c) * num_heads + head_idx
            gathered = jnp.take(value_flat, flat.reshape(-1), axis=0)
            gathered = gathered.reshape(num_queries, num_heads, num_samples, head_dim)
            out = out + jnp.sum(
                gathered * weight.reshape(num_queries, num_heads, num_samples, 1), axis=2
            )

        return out

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

    def _build_case(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        sampling_loc: torch.Tensor,
        attn_weight: torch.Tensor,
        num_queries: int,
        num_heads: int,
        head_dim: int,
        num_levels: int,
        num_points: int,
    ) -> Dict[str, Any]:
        return {
            "value": value,
            "spatial_shapes": spatial_shapes,
            "sampling_loc": sampling_loc,
            "attn_weight": attn_weight,
            "output": torch.empty(
                (num_queries, num_heads, head_dim), device=self.device, dtype=torch.float32
            ),
            "num_queries": num_queries,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_levels": num_levels,
            "num_points": num_points,
        }

    def _random_case(
        self,
        shapes: List[tuple],
        num_queries: int,
        num_heads: int,
        head_dim: int,
        num_points: int,
        loc_range: tuple = (0.0, 1.0),
        value_range: tuple = (-1.0, 1.0),
        zero_value: bool = False,
    ) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        num_levels = len(shapes)
        spatial_shapes = torch.tensor(shapes, device=device, dtype=torch.int32)
        num_rows = sum(h * w for h, w in shapes)

        if zero_value:
            value = torch.zeros((num_rows, num_heads, head_dim), device=device, dtype=dtype)
        else:
            value = torch.empty((num_rows, num_heads, head_dim), device=device, dtype=dtype)
            value.uniform_(value_range[0], value_range[1])

        sampling_loc = torch.empty(
            (num_queries, num_heads, num_levels, num_points, 2), device=device, dtype=dtype
        )
        sampling_loc.uniform_(loc_range[0], loc_range[1])

        logits = torch.empty(
            (num_queries, num_heads, num_levels * num_points), device=device, dtype=dtype
        )
        logits.uniform_(-2.0, 2.0)
        attn_weight = torch.softmax(logits, dim=-1).reshape(
            num_queries, num_heads, num_levels, num_points
        )

        return self._build_case(
            value,
            spatial_shapes,
            sampling_loc,
            attn_weight,
            num_queries,
            num_heads,
            head_dim,
            num_levels,
            num_points,
        )

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        spatial_shapes = torch.tensor([[2, 2], [1, 1]], device=device, dtype=torch.int32)
        value = torch.tensor(
            [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]], [[9.0, 10.0]]],
            device=device,
            dtype=dtype,
        )
        sampling_loc = torch.tensor(
            [
                [[[[0.25, 0.25]], [[0.5, 0.5]]]],
                [[[[0.50, 0.50]], [[0.5, 0.5]]]],
            ],
            device=device,
            dtype=dtype,
        )
        attn_weight = torch.tensor(
            [
                [[[0.5], [0.5]]],
                [[[0.25], [0.75]]],
            ],
            device=device,
            dtype=dtype,
        )
        return self._build_case(value, spatial_shapes, sampling_loc, attn_weight, 2, 1, 2, 2, 1)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(0)
        tests: List[Dict[str, Any]] = []

        # smallest possible problem: one query, one head, one 1x1 level, one point
        tests.append(
            self._random_case([(1, 1)], num_queries=1, num_heads=1, head_dim=1, num_points=1)
        )

        # two queries, sampling locations that spill outside every feature map
        tests.append(
            self._random_case(
                [(2, 3), (1, 1)],
                num_queries=2,
                num_heads=2,
                head_dim=4,
                num_points=2,
                loc_range=(-0.3, 1.3),
            )
        )

        # all-zero value buffer
        tests.append(
            self._random_case(
                [(4, 4), (2, 2), (1, 1)],
                num_queries=3,
                num_heads=1,
                head_dim=8,
                num_points=4,
                zero_value=True,
            )
        )

        # strictly negative features, non-square levels
        tests.append(
            self._random_case(
                [(5, 7), (3, 3)],
                num_queries=4,
                num_heads=4,
                head_dim=8,
                num_points=3,
                value_range=(-3.0, -0.5),
            )
        )

        # every sampling location far outside the feature maps -> all samples are zero
        tests.append(
            self._random_case(
                [(6, 6), (3, 3)],
                num_queries=4,
                num_heads=2,
                head_dim=16,
                num_points=2,
                loc_range=(2.0, 4.0),
            )
        )

        # power-of-two sizes with the usual Deformable DETR head configuration
        tests.append(
            self._random_case(
                [(16, 16), (8, 8), (4, 4), (2, 2)],
                num_queries=16,
                num_heads=8,
                head_dim=32,
                num_points=4,
            )
        )

        # non-power-of-two everything
        tests.append(
            self._random_case(
                [(13, 17), (7, 9), (3, 5)],
                num_queries=100,
                num_heads=4,
                head_dim=16,
                num_points=4,
            )
        )

        # non-power-of-two with out-of-range locations
        tests.append(
            self._random_case(
                [(30, 30), (15, 15), (8, 8), (4, 4)],
                num_queries=255,
                num_heads=8,
                head_dim=32,
                num_points=4,
                loc_range=(-0.2, 1.2),
            )
        )

        # larger power-of-two pyramid
        tests.append(
            self._random_case(
                [(32, 32), (16, 16), (8, 8), (4, 4)],
                num_queries=1024,
                num_heads=8,
                head_dim=32,
                num_points=4,
            )
        )

        # realistic detection-decoder workload
        tests.append(
            self._random_case(
                [(64, 64), (32, 32), (16, 16), (8, 8)],
                num_queries=4000,
                num_heads=8,
                head_dim=32,
                num_points=4,
            )
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # encoder self-attention over a 4-level feature pyramid: every pixel is a query
        return self._random_case(
            [(96, 160), (48, 80), (24, 40), (12, 20)],
            num_queries=20400,
            num_heads=8,
            head_dim=32,
            num_points=4,
        )
