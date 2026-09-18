import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Vision Transformer Patch Embedding"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        images: torch.Tensor,
        patch_weight: torch.Tensor,
        patch_bias: torch.Tensor,
        cls_token: torch.Tensor,
        pos_embed: torch.Tensor,
        output: torch.Tensor,
        B: int,
        C: int,
        H: int,
        W: int,
        P: int,
        D: int,
    ):
        gh, gw = H // P, W // P
        N = gh * gw
        assert images.shape == (B, C, H, W)
        assert patch_weight.shape == (D, C, P, P)
        assert patch_bias.shape == (D,)
        assert cls_token.shape == (D,)
        assert pos_embed.shape == (N + 1, D)
        assert output.shape == (B, N + 1, D)
        assert (
            images.dtype
            == patch_weight.dtype
            == patch_bias.dtype
            == cls_token.dtype
            == pos_embed.dtype
            == output.dtype
            == torch.float32
        )

        # (B, C, gh, P, gw, P) -> (B, gh, gw, C, P, P) -> (B, N, C * P * P)
        patches = images.view(B, C, gh, P, gw, P).permute(0, 2, 4, 1, 3, 5).reshape(B, N, C * P * P)
        tokens = patches @ patch_weight.view(D, C * P * P).t() + patch_bias  # (B, N, D)
        cls = cls_token.view(1, 1, D).expand(B, 1, D)
        output.copy_(torch.cat([cls, tokens], dim=1) + pos_embed)

    def reference_impl_jax(
        self, images, patch_weight, patch_bias, cls_token, pos_embed, B, C, H, W, P, D
    ):
        import jax.numpy as jnp

        gh, gw = H // P, W // P
        N = gh * gw
        patches = (
            images.reshape(B, C, gh, P, gw, P).transpose(0, 2, 4, 1, 3, 5).reshape(B, N, C * P * P)
        )
        tokens = patches @ patch_weight.reshape(D, C * P * P).T + patch_bias
        cls = jnp.broadcast_to(cls_token.reshape(1, 1, D), (B, 1, D))
        return jnp.concatenate([cls, tokens], axis=1) + pos_embed

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "images": (ctypes.POINTER(ctypes.c_float), "in"),
            "patch_weight": (ctypes.POINTER(ctypes.c_float), "in"),
            "patch_bias": (ctypes.POINTER(ctypes.c_float), "in"),
            "cls_token": (ctypes.POINTER(ctypes.c_float), "in"),
            "pos_embed": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "W": (ctypes.c_int, "in"),
            "P": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def _make_test_case(self, B, C, H, W, P, D, fill=None):
        device = self.device
        dtype = torch.float32
        N = (H // P) * (W // P)
        if fill is None:
            images = torch.randn(B, C, H, W, device=device, dtype=dtype)
        else:
            images = torch.full((B, C, H, W), fill, device=device, dtype=dtype)
        return {
            "images": images,
            "patch_weight": torch.randn(D, C, P, P, device=device, dtype=dtype) * 0.02,
            "patch_bias": torch.randn(D, device=device, dtype=dtype) * 0.02,
            "cls_token": torch.randn(D, device=device, dtype=dtype) * 0.02,
            "pos_embed": torch.randn(N + 1, D, device=device, dtype=dtype) * 0.02,
            "output": torch.empty(B, N + 1, D, device=device, dtype=dtype),
            "B": B,
            "C": C,
            "H": H,
            "W": W,
            "P": P,
            "D": D,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32
        B, C, H, W, P, D = 1, 1, 4, 4, 2, 2
        images = torch.tensor(
            [
                [
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        # filter 0 selects the top-left pixel of a patch, filter 1 the bottom-right pixel
        patch_weight = torch.tensor(
            [[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 1.0]]]],
            device=device,
            dtype=dtype,
        )
        patch_bias = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
        cls_token = torch.tensor([0.5, -0.5], device=device, dtype=dtype)
        pos_embed = torch.tensor(
            [[0.0, 0.0], [1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]],
            device=device,
            dtype=dtype,
        )
        return {
            "images": images,
            "patch_weight": patch_weight,
            "patch_bias": patch_bias,
            "cls_token": cls_token,
            "pos_embed": pos_embed,
            "output": torch.empty(B, 5, D, device=device, dtype=dtype),
            "B": B,
            "C": C,
            "H": H,
            "W": W,
            "P": P,
            "D": D,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge case: a single image that is exactly one patch
        tests.append(self._make_test_case(1, 1, 2, 2, 2, 4))

        # Edge case: tiny multi-channel image, 4 patches
        tests.append(self._make_test_case(2, 3, 4, 4, 2, 8))

        # Zero input: only bias, cls token and positional embedding survive
        tests.append(self._make_test_case(1, 3, 8, 8, 4, 16, fill=0.0))

        # Negative-only input
        tests.append(self._make_test_case(2, 3, 16, 16, 4, 32, fill=-1.5))

        # Power-of-2 sizes
        tests.append(self._make_test_case(4, 3, 32, 32, 16, 64))
        tests.append(self._make_test_case(2, 3, 64, 64, 8, 128))

        # Non-power-of-2 sizes
        tests.append(self._make_test_case(3, 3, 30, 30, 5, 100))

        # Non-square image with a non-power-of-2 patch size
        tests.append(self._make_test_case(5, 1, 18, 24, 6, 48))

        # Realistic: ViT-Tiny/16 at 224x224
        tests.append(self._make_test_case(8, 3, 224, 224, 16, 192))

        # Realistic: small patches, wide embedding
        tests.append(self._make_test_case(2, 3, 96, 96, 8, 256))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # ViT-Base/16 at 224x224 with a batch of 64 images
        torch.manual_seed(0)
        device = self.device
        dtype = torch.float32
        B, C, H, W, P, D = 64, 3, 224, 224, 16, 768
        N = (H // P) * (W // P)
        return {
            "images": torch.empty(B, C, H, W, device=device, dtype=dtype).uniform_(-1.0, 1.0),
            "patch_weight": torch.randn(D, C, P, P, device=device, dtype=dtype) * 0.02,
            "patch_bias": torch.randn(D, device=device, dtype=dtype) * 0.02,
            "cls_token": torch.randn(D, device=device, dtype=dtype) * 0.02,
            "pos_embed": torch.randn(N + 1, D, device=device, dtype=dtype) * 0.02,
            "output": torch.empty(B, N + 1, D, device=device, dtype=dtype),
            "B": B,
            "C": C,
            "H": H,
            "W": W,
            "P": P,
            "D": D,
        }
