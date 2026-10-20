import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandTensor


class Challenge(ChallengeBase):
    name = "Gated Attention"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        G: torch.Tensor,
        output: torch.Tensor,
        M: int,
        d: int,
    ):
        assert Q.shape == K.shape == V.shape == G.shape == output.shape == (M, d)
        assert Q.dtype == K.dtype == V.dtype == G.dtype == output.dtype

        scores = (Q @ K.T) / (d**0.5)

        idxs = torch.arange(M, device=Q.device)
        is_future = idxs.unsqueeze(0) > idxs.unsqueeze(1)
        scores = scores.masked_fill(is_future, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn_output = attn @ V

        # Gated attention (Qiu et al., 2025): a sigmoid gate, produced from the
        # gate logits G, is applied element-wise to the SDPA output.
        torch.mul(attn_output, torch.sigmoid(G), out=output)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "G": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        K = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        V = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        G = torch.tensor(
            [
                [2.0, 0.0, -2.0, 1.0],
                [0.0, 2.0, 1.0, -1.0],
                [-1.0, 1.0, 2.0, 0.0],
                [1.0, -2.0, 0.0, 2.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        output = torch.empty(4, 4, device=self.device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "G": G, "output": output, "M": 4, "d": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_example (matches example)
        tests.append(
            {
                "Q": torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "K": torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "G": torch.tensor(
                    [
                        [2.0, 0.0, -2.0, 1.0],
                        [0.0, 2.0, 1.0, -1.0],
                        [-1.0, 1.0, 2.0, 0.0],
                        [1.0, -2.0, 0.0, 2.0],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.empty(4, 4, device=self.device, dtype=dtype),
                "M": 4,
                "d": 4,
            }
        )

        # single-token edge case: softmax over one key is 1, so the output is
        # exactly V * sigmoid(G) = [1.867378, -0.268941]
        tests.append(
            {
                "Q": torch.tensor([[1.0, 0.0]], device=self.device, dtype=dtype),
                "K": torch.tensor([[0.0, 1.0]], device=self.device, dtype=dtype),
                "V": torch.tensor([[3.0, -1.0]], device=self.device, dtype=dtype),
                "G": torch.tensor([[0.5, -1.0]], device=self.device, dtype=dtype),
                "output": torch.empty(1, 2, device=self.device, dtype=dtype),
                "M": 1,
                "d": 2,
            }
        )

        # two-token case with zero gate logits: every gate is exactly 0.5, which
        # catches implementations that skip the sigmoid (gate factor would be 0)
        tests.append(
            {
                "Q": torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device, dtype=dtype),
                "K": torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device, dtype=dtype),
                "V": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device, dtype=dtype),
                "G": torch.zeros((2, 2), device=self.device, dtype=dtype),
                "output": torch.empty(2, 2, device=self.device, dtype=dtype),
                "M": 2,
                "d": 2,
            }
        )

        # zero matrices
        tests.append(
            {
                "Q": torch.zeros((6, 8), device=self.device, dtype=dtype),
                "K": torch.zeros((6, 8), device=self.device, dtype=dtype),
                "V": torch.zeros((6, 8), device=self.device, dtype=dtype),
                "G": torch.zeros((6, 8), device=self.device, dtype=dtype),
                "output": torch.empty(6, 8, device=self.device, dtype=dtype),
                "M": 6,
                "d": 8,
            }
        )

        # gate fully closed: sigmoid(-50) is ~0, so the output must be ~0 even
        # though plain causal attention would return large values. Catches
        # implementations that forget to apply the gate at all.
        M, d = 30, 16
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "G": torch.full((M, d), -50.0, device=self.device, dtype=dtype),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
            }
        )

        # gate fully open: sigmoid(50) is ~1, so the output must equal plain
        # causal attention. Together with the closed-gate case, this catches an
        # inverted gate such as sigmoid(-G).
        M, d = 30, 16
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "G": torch.full((M, d), 50.0, device=self.device, dtype=dtype),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
            }
        )

        # mixed values with negatives
        tests.append(
            {
                "Q": torch.tensor(
                    [
                        [-1.0, 2.0, -3.0, 0.5],
                        [4.0, -5.0, 6.0, -0.5],
                        [-7.0, 8.0, -9.0, 1.0],
                        [10.0, -11.0, 12.0, -1.0],
                        [1.0, 1.0, -2.0, 3.0],
                        [-3.0, 2.0, 1.0, -1.0],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "K": torch.tensor(
                    [
                        [2.0, -1.0, 3.0, 0.25],
                        [-4.0, 5.0, -6.0, 0.75],
                        [7.0, -8.0, 9.0, -0.25],
                        [-10.0, 11.0, -12.0, 0.5],
                        [1.0, -2.0, 3.0, -3.0],
                        [-1.0, 2.0, -3.0, 3.0],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [
                        [1.0, 0.5, -0.5, 2.0],
                        [-1.0, 2.0, 3.0, -2.0],
                        [4.0, -2.0, 1.0, 0.0],
                        [0.0, 1.0, -1.0, 3.0],
                        [2.0, 3.0, -3.0, 1.0],
                        [-2.0, -3.0, 3.0, -1.0],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "G": torch.tensor(
                    [
                        [1.0, -1.0, 2.0, -2.0],
                        [-3.0, 0.5, 1.5, 0.0],
                        [2.0, 2.0, -1.0, -0.5],
                        [0.0, -2.0, 3.0, 1.0],
                        [-1.5, 1.0, 0.0, 2.5],
                        [3.0, -3.0, -2.0, 0.5],
                    ],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.empty(6, 4, device=self.device, dtype=dtype),
                "M": 6,
                "d": 4,
            }
        )

        # power-of-two dimensions
        M, d = 64, 32
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "G": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-3.0, 3.0),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
            }
        )

        # non-power-of-two dimensions
        M, d = 100, 48
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "G": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-3.0, 3.0),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
            }
        )

        # realistic size
        M, d = 512, 64
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.2, 0.2),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.2, 0.2),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "G": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M, d = 4096, 128
        return {
            "Q": RandTensor((M, d), -1.0, 1.0),
            "K": RandTensor((M, d), -1.0, 1.0),
            "V": RandTensor((M, d), -1.0, 1.0),
            "G": RandTensor((M, d), -3.0, 3.0),
            "output": OutTensor((M, d)),
            "M": M,
            "d": d,
        }
