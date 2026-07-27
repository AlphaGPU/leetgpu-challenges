import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Softmax Attention Backward"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        dO: torch.Tensor,
        dQ: torch.Tensor,
        dK: torch.Tensor,
        dV: torch.Tensor,
        M: int,
        N: int,
        d: int,
    ):
        assert Q.shape == (M, d)
        assert K.shape == (N, d)
        assert V.shape == (N, d)
        assert dO.shape == (M, d)
        assert dQ.shape == (M, d)
        assert dK.shape == (N, d)
        assert dV.shape == (N, d)
        assert (
            Q.dtype
            == K.dtype
            == V.dtype
            == dO.dtype
            == dQ.dtype
            == dK.dtype
            == dV.dtype
            == torch.float32
        )

        scale = d**0.5
        S = torch.matmul(Q, K.t()) / scale
        P = torch.softmax(S, dim=1)
        torch.matmul(P.t(), dO, out=dV)
        dP = torch.matmul(dO, V.t())
        dS = dP * P - P * (dP * P).sum(dim=1, keepdim=True)
        dS = dS / scale
        torch.matmul(dS, K, out=dQ)
        torch.matmul(dS.t(), Q, out=dK)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "dO": (ctypes.POINTER(ctypes.c_float), "in"),
            "dQ": (ctypes.POINTER(ctypes.c_float), "out"),
            "dK": (ctypes.POINTER(ctypes.c_float), "out"),
            "dV": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
        )
        K = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            device=self.device,
            dtype=dtype,
        )
        V = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            device=self.device,
            dtype=dtype,
        )
        dO = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
        )
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "dO": dO,
            "dQ": torch.empty(2, 4, device=self.device, dtype=dtype),
            "dK": torch.empty(3, 4, device=self.device, dtype=dtype),
            "dV": torch.empty(3, 4, device=self.device, dtype=dtype),
            "M": 2,
            "N": 3,
            "d": 4,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # minimal_edge_case
        tests.append(
            {
                "Q": torch.tensor([[1.0]], device=self.device, dtype=dtype),
                "K": torch.tensor([[1.0]], device=self.device, dtype=dtype),
                "V": torch.tensor([[2.0]], device=self.device, dtype=dtype),
                "dO": torch.tensor([[1.0]], device=self.device, dtype=dtype),
                "dQ": torch.empty(1, 1, device=self.device, dtype=dtype),
                "dK": torch.empty(1, 1, device=self.device, dtype=dtype),
                "dV": torch.empty(1, 1, device=self.device, dtype=dtype),
                "M": 1,
                "N": 1,
                "d": 1,
            }
        )

        # basic_example
        tests.append(
            {
                "Q": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
                ),
                "K": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "dO": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=self.device, dtype=dtype
                ),
                "dQ": torch.empty(2, 4, device=self.device, dtype=dtype),
                "dK": torch.empty(3, 4, device=self.device, dtype=dtype),
                "dV": torch.empty(3, 4, device=self.device, dtype=dtype),
                "M": 2,
                "N": 3,
                "d": 4,
            }
        )

        # zero_inputs
        tests.append(
            {
                "Q": torch.zeros((3, 5), device=self.device, dtype=dtype),
                "K": torch.zeros((4, 5), device=self.device, dtype=dtype),
                "V": torch.zeros((4, 5), device=self.device, dtype=dtype),
                "dO": torch.zeros((3, 5), device=self.device, dtype=dtype),
                "dQ": torch.empty(3, 5, device=self.device, dtype=dtype),
                "dK": torch.empty(4, 5, device=self.device, dtype=dtype),
                "dV": torch.empty(4, 5, device=self.device, dtype=dtype),
                "M": 3,
                "N": 4,
                "d": 5,
            }
        )

        # negative_and_mixed_values
        tests.append(
            {
                "Q": torch.tensor(
                    [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "K": torch.tensor(
                    [[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "dO": torch.tensor(
                    [[-0.5, 1.0, 0.5], [0.5, -1.0, 0.5], [1.0, 0.5, -0.5], [-0.5, 0.5, 1.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "dQ": torch.empty(4, 3, device=self.device, dtype=dtype),
                "dK": torch.empty(4, 3, device=self.device, dtype=dtype),
                "dV": torch.empty(4, 3, device=self.device, dtype=dtype),
                "M": 4,
                "N": 4,
                "d": 3,
            }
        )

        # power_of_2_small
        tests.append(
            {
                "Q": torch.randn((16, 8), device=self.device, dtype=dtype),
                "K": torch.randn((16, 8), device=self.device, dtype=dtype),
                "V": torch.randn((16, 8), device=self.device, dtype=dtype),
                "dO": torch.randn((16, 8), device=self.device, dtype=dtype),
                "dQ": torch.empty(16, 8, device=self.device, dtype=dtype),
                "dK": torch.empty(16, 8, device=self.device, dtype=dtype),
                "dV": torch.empty(16, 8, device=self.device, dtype=dtype),
                "M": 16,
                "N": 16,
                "d": 8,
            }
        )

        # power_of_2_medium
        tests.append(
            {
                "Q": torch.randn((64, 32), device=self.device, dtype=dtype),
                "K": torch.randn((128, 32), device=self.device, dtype=dtype),
                "V": torch.randn((128, 32), device=self.device, dtype=dtype),
                "dO": torch.randn((64, 32), device=self.device, dtype=dtype),
                "dQ": torch.empty(64, 32, device=self.device, dtype=dtype),
                "dK": torch.empty(128, 32, device=self.device, dtype=dtype),
                "dV": torch.empty(128, 32, device=self.device, dtype=dtype),
                "M": 64,
                "N": 128,
                "d": 32,
            }
        )

        # non_power_of_2_small
        tests.append(
            {
                "Q": torch.randn((30, 7), device=self.device, dtype=dtype),
                "K": torch.randn((15, 7), device=self.device, dtype=dtype),
                "V": torch.randn((15, 7), device=self.device, dtype=dtype),
                "dO": torch.randn((30, 7), device=self.device, dtype=dtype),
                "dQ": torch.empty(30, 7, device=self.device, dtype=dtype),
                "dK": torch.empty(15, 7, device=self.device, dtype=dtype),
                "dV": torch.empty(15, 7, device=self.device, dtype=dtype),
                "M": 30,
                "N": 15,
                "d": 7,
            }
        )

        # non_power_of_2_large
        tests.append(
            {
                "Q": torch.randn((100, 50), device=self.device, dtype=dtype),
                "K": torch.randn((255, 50), device=self.device, dtype=dtype),
                "V": torch.randn((255, 50), device=self.device, dtype=dtype),
                "dO": torch.randn((100, 50), device=self.device, dtype=dtype),
                "dQ": torch.empty(100, 50, device=self.device, dtype=dtype),
                "dK": torch.empty(255, 50, device=self.device, dtype=dtype),
                "dV": torch.empty(255, 50, device=self.device, dtype=dtype),
                "M": 100,
                "N": 255,
                "d": 50,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N, d = 512, 256, 128
        Q = torch.randn((M, d), device=self.device, dtype=dtype)
        K = torch.randn((N, d), device=self.device, dtype=dtype)
        V = torch.randn((N, d), device=self.device, dtype=dtype)
        dO = torch.randn((M, d), device=self.device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "dO": dO,
            "dQ": torch.empty(M, d, device=self.device, dtype=dtype),
            "dK": torch.empty(N, d, device=self.device, dtype=dtype),
            "dV": torch.empty(N, d, device=self.device, dtype=dtype),
            "M": M,
            "N": N,
            "d": d,
        }
