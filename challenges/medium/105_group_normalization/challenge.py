import ctypes
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Group Normalization"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        X: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        Y: torch.Tensor,
        N: int,
        C: int,
        H: int,
        W: int,
        G: int,
        eps: float,
    ):
        assert X.shape == Y.shape == (N, C, H, W)
        assert gamma.shape == beta.shape == (C,)
        assert X.dtype == gamma.dtype == beta.dtype == Y.dtype
        assert C % G == 0

        Y.copy_(F.group_norm(X, G, gamma, beta, eps=eps))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "Y": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in"),
            "H": (ctypes.c_int, "in"),
            "W": (ctypes.c_int, "in"),
            "G": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, C, H, W, G = 1, 4, 2, 2, 2
        X = torch.tensor(
            [
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[3.0, 3.0], [3.0, 3.0]],
                    [[2.0, 2.0], [2.0, 2.0]],
                    [[6.0, 6.0], [6.0, 6.0]],
                ]
            ],
            device=self.device,
            dtype=dtype,
        )
        gamma = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype)
        beta = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype)
        Y = torch.empty((N, C, H, W), device=self.device, dtype=dtype)
        return {
            "X": X,
            "gamma": gamma,
            "beta": beta,
            "Y": Y,
            "N": N,
            "C": C,
            "H": H,
            "W": W,
            "G": G,
            "eps": 1e-5,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_small: matches example shape
        N, C, H, W, G = 1, 4, 2, 2, 2
        tests.append(
            {
                "X": torch.tensor(
                    [
                        [
                            [[1.0, 1.0], [1.0, 1.0]],
                            [[3.0, 3.0], [3.0, 3.0]],
                            [[2.0, 2.0], [2.0, 2.0]],
                            [[6.0, 6.0], [6.0, 6.0]],
                        ]
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "gamma": torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype),
                "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "H": H,
                "W": W,
                "G": G,
                "eps": 1e-5,
            }
        )

        # layernorm_like: G=1 reduces over (C, H, W)
        N, C, H, W, G = 2, 8, 4, 4, 1
        tests.append(
            {
                "X": torch.empty((N, C, H, W), device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.5, 1.5),
                "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "H": H,
                "W": W,
                "G": G,
                "eps": 1e-5,
            }
        )

        # instancenorm_like: G=C, each channel is its own group
        N, C, H, W, G = 2, 4, 3, 3, 4
        tests.append(
            {
                "X": torch.empty((N, C, H, W), device=self.device, dtype=dtype).uniform_(-3.0, 3.0),
                "gamma": torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype),
                "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "H": H,
                "W": W,
                "G": G,
                "eps": 1e-5,
            }
        )

        # all_zeros
        N, C, H, W, G = 2, 8, 4, 4, 2
        tests.append(
            {
                "X": torch.zeros((N, C, H, W), device=self.device, dtype=dtype),
                "gamma": torch.ones(C, device=self.device, dtype=dtype),
                "beta": torch.zeros(C, device=self.device, dtype=dtype),
                "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "H": H,
                "W": W,
                "G": G,
                "eps": 1e-5,
            }
        )

        # negative_numbers and nontrivial gamma/beta
        N, C, H, W, G = 2, 6, 4, 4, 3
        tests.append(
            {
                "X": torch.empty((N, C, H, W), device=self.device, dtype=dtype).uniform_(-5.0, 0.0),
                "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.5, 2.0),
                "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "H": H,
                "W": W,
                "G": G,
                "eps": 1e-5,
            }
        )

        # non_power_of_2: C=12, G=4 (3 channels per group), H=W=7
        N, C, H, W, G = 3, 12, 7, 7, 4
        tests.append(
            {
                "X": torch.empty((N, C, H, W), device=self.device, dtype=dtype).uniform_(
                    -10.0, 10.0
                ),
                "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.5, 2.0),
                "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "H": H,
                "W": W,
                "G": G,
                "eps": 1e-5,
            }
        )

        # mid_size: 32 channels, 16x16 spatial
        N, C, H, W, G = 4, 32, 16, 16, 8
        tests.append(
            {
                "X": torch.empty((N, C, H, W), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.5, 1.5),
                "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "H": H,
                "W": W,
                "G": G,
                "eps": 1e-5,
            }
        )

        # large_realistic: 64 channels, 32x32 spatial
        N, C, H, W, G = 2, 64, 32, 32, 16
        tests.append(
            {
                "X": torch.empty((N, C, H, W), device=self.device, dtype=dtype).uniform_(-5.0, 5.0),
                "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.5, 1.5),
                "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "H": H,
                "W": W,
                "G": G,
                "eps": 1e-5,
            }
        )

        # non_square_spatial: H != W
        N, C, H, W, G = 2, 16, 8, 32, 4
        tests.append(
            {
                "X": torch.empty((N, C, H, W), device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.8, 1.2),
                "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
                "N": N,
                "C": C,
                "H": H,
                "W": W,
                "G": G,
                "eps": 1e-5,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, C, H, W, G = 8, 512, 64, 64, 32
        return {
            "X": torch.empty((N, C, H, W), device=self.device, dtype=dtype).uniform_(-3.0, 3.0),
            "gamma": torch.empty(C, device=self.device, dtype=dtype).uniform_(0.5, 1.5),
            "beta": torch.empty(C, device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
            "Y": torch.empty((N, C, H, W), device=self.device, dtype=dtype),
            "N": N,
            "C": C,
            "H": H,
            "W": W,
            "G": G,
            "eps": 1e-5,
        }
