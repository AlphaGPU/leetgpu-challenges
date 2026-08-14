import ctypes
from typing import Any, Dict, List, Sequence, Tuple

import torch
from core.challenge_base import ChallengeBase


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
        N: int,
        S: int,
        Q: int,
        H: int,
        D: int,
        L: int,
        P: int,
    ):
        assert value.shape == (N, S, H, D)
        assert spatial_shapes.shape == (L, 2)
        assert sampling_loc.shape == (N, Q, H, L, P, 2)
        assert attn_weight.shape == (N, Q, H, L, P)
        assert output.shape == (N, Q, H * D)
        assert value.dtype == sampling_loc.dtype == attn_weight.dtype == output.dtype
        assert spatial_shapes.dtype == torch.int32

        device = value.device
        dtype = value.dtype
        shapes = spatial_shapes.tolist()

        acc = torch.zeros((N, Q, H, D), device=device, dtype=dtype)
        n_idx = torch.arange(N, device=device).view(N, 1, 1, 1)
        h_idx = torch.arange(H, device=device).view(1, 1, H, 1)
        zero = torch.zeros((), device=device, dtype=dtype)

        level_start = 0
        for level, (h_l, w_l) in enumerate(shapes):
            value_l = value[:, level_start : level_start + h_l * w_l]
            loc = sampling_loc[:, :, :, level]
            weight = attn_weight[:, :, :, level]

            px = loc[..., 0] * w_l - 0.5
            py = loc[..., 1] * h_l - 0.5
            x0 = torch.floor(px)
            y0 = torch.floor(py)
            dx = px - x0
            dy = py - y0

            for cy in (0, 1):
                for cx in (0, 1):
                    xi = x0 + cx
                    yi = y0 + cy
                    valid = (xi >= 0) & (xi <= w_l - 1) & (yi >= 0) & (yi <= h_l - 1)
                    flat = yi.clamp(0, h_l - 1).long() * w_l + xi.clamp(0, w_l - 1).long()
                    gathered = value_l[n_idx, flat, h_idx]

                    corner_w = (dx if cx else 1.0 - dx) * (dy if cy else 1.0 - dy)
                    coef = torch.where(valid, corner_w, zero) * weight
                    acc += (coef.unsqueeze(-1) * gathered).sum(dim=3)

            level_start += h_l * w_l

        output.copy_(acc.reshape(N, Q, H * D))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "value": (ctypes.POINTER(ctypes.c_float), "in"),
            "spatial_shapes": (ctypes.POINTER(ctypes.c_int), "in"),
            "sampling_loc": (ctypes.POINTER(ctypes.c_float), "in"),
            "attn_weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
            "Q": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "L": (ctypes.c_int, "in"),
            "P": (ctypes.c_int, "in"),
        }

    def _random_case(
        self,
        N: int,
        Q: int,
        H: int,
        D: int,
        shapes: Sequence[Tuple[int, int]],
        P: int,
        loc_range: Tuple[float, float] = (0.0, 1.0),
        value_range: Tuple[float, float] = (-1.0, 1.0),
        zero_value: bool = False,
    ) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        L = len(shapes)
        S = sum(h * w for h, w in shapes)

        spatial_shapes = torch.tensor(shapes, device=device, dtype=torch.int32)
        if zero_value:
            value = torch.zeros((N, S, H, D), device=device, dtype=dtype)
        else:
            value = torch.empty((N, S, H, D), device=device, dtype=dtype).uniform_(
                value_range[0], value_range[1]
            )
        sampling_loc = torch.empty((N, Q, H, L, P, 2), device=device, dtype=dtype).uniform_(
            loc_range[0], loc_range[1]
        )
        weight = torch.empty((N, Q, H, L, P), device=device, dtype=dtype).uniform_(0.05, 1.0)
        attn_weight = weight / weight.sum(dim=(3, 4), keepdim=True)

        return {
            "value": value,
            "spatial_shapes": spatial_shapes,
            "sampling_loc": sampling_loc,
            "attn_weight": attn_weight,
            "output": torch.empty((N, Q, H * D), device=device, dtype=dtype),
            "N": N,
            "S": S,
            "Q": Q,
            "H": H,
            "D": D,
            "L": L,
            "P": P,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        N, Q, H, D, L, P = 1, 1, 1, 2, 2, 1
        shapes = [(2, 2), (1, 1)]
        S = 5

        value = torch.tensor(
            [[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]], [[10.0, 20.0]]]],
            device=device,
            dtype=dtype,
        )
        sampling_loc = torch.tensor(
            [[[[[[0.5, 0.5]], [[0.5, 0.5]]]]]],
            device=device,
            dtype=dtype,
        )
        attn_weight = torch.tensor([[[[[0.75], [0.25]]]]], device=device, dtype=dtype)

        return {
            "value": value,
            "spatial_shapes": torch.tensor(shapes, device=device, dtype=torch.int32),
            "sampling_loc": sampling_loc,
            "attn_weight": attn_weight,
            "output": torch.empty((N, Q, H * D), device=device, dtype=dtype),
            "N": N,
            "S": S,
            "Q": Q,
            "H": H,
            "D": D,
            "L": L,
            "P": P,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests: List[Dict[str, Any]] = []

        # smallest possible problem: one query, one head, one point, 1x1 feature map
        tests.append(self._random_case(N=1, Q=1, H=1, D=1, shapes=[(1, 1)], P=1))

        # all-zero value tensor
        tests.append(
            self._random_case(N=2, Q=3, H=2, D=4, shapes=[(3, 3), (2, 2)], P=2, zero_value=True)
        )

        # tiny multi-level case with strongly negative features
        tests.append(
            self._random_case(
                N=1, Q=4, H=2, D=3, shapes=[(2, 3), (1, 2)], P=4, value_range=(-4.0, -0.5)
            )
        )

        # sampling locations far outside [0, 1] exercise zero padding
        tests.append(
            self._random_case(
                N=2, Q=16, H=4, D=8, shapes=[(4, 4), (2, 2)], P=4, loc_range=(-0.6, 1.6)
            )
        )

        # power-of-two dimensions
        tests.append(self._random_case(N=2, Q=64, H=8, D=32, shapes=[(8, 8), (4, 4), (2, 2)], P=4))

        # non-power-of-two dimensions with non-square feature maps
        tests.append(self._random_case(N=3, Q=30, H=3, D=17, shapes=[(7, 11), (5, 3), (3, 2)], P=3))

        # single level, many sampling points per query
        tests.append(self._random_case(N=1, Q=100, H=5, D=24, shapes=[(13, 17)], P=8))

        # five levels, one point each
        tests.append(
            self._random_case(
                N=2, Q=255, H=4, D=16, shapes=[(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)], P=1
            )
        )

        # realistic detection-style workload
        tests.append(
            self._random_case(
                N=2, Q=1024, H=8, D=32, shapes=[(32, 32), (16, 16), (8, 8), (4, 4)], P=4
            )
        )

        # larger realistic workload with out-of-range locations mixed in
        tests.append(
            self._random_case(
                N=1,
                Q=4000,
                H=8,
                D=32,
                shapes=[(64, 64), (32, 32), (16, 16), (8, 8)],
                P=4,
                loc_range=(-0.1, 1.1),
            )
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._random_case(
            N=2,
            Q=20000,
            H=8,
            D=32,
            shapes=[(128, 128), (64, 64), (32, 32), (16, 16)],
            P=4,
        )
