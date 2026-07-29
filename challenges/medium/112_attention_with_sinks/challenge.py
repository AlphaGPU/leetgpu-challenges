import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandTensor


class Challenge(ChallengeBase):
    name = "Attention with Sinks"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        M: int,
        d: int,
        num_sinks: int,
        window_size: int,
    ):
        assert Q.shape == K.shape == V.shape == output.shape == (M, d)
        assert Q.dtype == K.dtype == V.dtype == output.dtype

        scores = (Q @ K.T) / (d**0.5)

        idxs = torch.arange(M, device=Q.device)
        i = idxs.unsqueeze(1)
        j = idxs.unsqueeze(0)
        is_causal = j <= i
        is_sink = j < num_sinks
        is_window = j >= (i - window_size + 1)
        allowed = is_causal & (is_sink | is_window)

        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        torch.matmul(attn, V, out=output)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "d": (ctypes.c_int, "in"),
            "num_sinks": (ctypes.c_int, "in"),
            "window_size": (ctypes.c_int, "in"),
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
        output = torch.empty(4, 4, device=self.device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "M": 4,
            "d": 4,
            "num_sinks": 1,
            "window_size": 2,
        }

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
                "output": torch.empty(4, 4, device=self.device, dtype=dtype),
                "M": 4,
                "d": 4,
                "num_sinks": 1,
                "window_size": 2,
            }
        )

        # single-token edge case
        tests.append(
            {
                "Q": torch.tensor([[0.5, -0.5, 1.0, 0.25]], device=self.device, dtype=dtype),
                "K": torch.tensor([[0.25, 1.0, -0.5, 0.5]], device=self.device, dtype=dtype),
                "V": torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device, dtype=dtype),
                "output": torch.empty(1, 4, device=self.device, dtype=dtype),
                "M": 1,
                "d": 4,
                "num_sinks": 1,
                "window_size": 1,
            }
        )

        # window covers entire prefix (equivalent to standard causal)
        tests.append(
            {
                "Q": torch.tensor(
                    [[1.0, -1.0, 0.5], [0.5, 0.5, -0.5], [-1.0, 1.0, 1.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "K": torch.tensor(
                    [[-0.5, 1.0, 0.5], [1.0, 0.0, -1.0], [0.5, -0.5, 1.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.empty(3, 3, device=self.device, dtype=dtype),
                "M": 3,
                "d": 3,
                "num_sinks": 1,
                "window_size": 3,
            }
        )

        # zero matrices with strict window
        tests.append(
            {
                "Q": torch.zeros((6, 8), device=self.device, dtype=dtype),
                "K": torch.zeros((6, 8), device=self.device, dtype=dtype),
                "V": torch.zeros((6, 8), device=self.device, dtype=dtype),
                "output": torch.empty(6, 8, device=self.device, dtype=dtype),
                "M": 6,
                "d": 8,
                "num_sinks": 2,
                "window_size": 2,
            }
        )

        # mixed values with negatives, sinks drop out of window
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
                "output": torch.empty(6, 4, device=self.device, dtype=dtype),
                "M": 6,
                "d": 4,
                "num_sinks": 2,
                "window_size": 2,
            }
        )

        # power-of-two dimensions
        M, d = 64, 32
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
                "num_sinks": 4,
                "window_size": 16,
            }
        )

        # non-power-of-two, wider heads
        M, d = 100, 48
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
                "num_sinks": 2,
                "window_size": 32,
            }
        )

        # window equals 1 (only current token + sinks)
        M, d = 255, 64
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.3, 0.3),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.3, 0.3),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
                "num_sinks": 4,
                "window_size": 1,
            }
        )

        # realistic size
        M, d = 512, 64
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.2, 0.2),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.2, 0.2),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
                "num_sinks": 4,
                "window_size": 128,
            }
        )

        # larger realistic
        M, d = 1024, 128
        tests.append(
            {
                "Q": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty((M, d), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.empty(M, d, device=self.device, dtype=dtype),
                "M": M,
                "d": d,
                "num_sinks": 8,
                "window_size": 256,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M, d, num_sinks, window_size = 5000, 128, 4, 1024
        return {
            "Q": RandTensor((M, d), -1.0, 1.0),
            "K": RandTensor((M, d), -1.0, 1.0),
            "V": RandTensor((M, d), -1.0, 1.0),
            "output": OutTensor((M, d)),
            "M": M,
            "d": d,
            "num_sinks": num_sinks,
            "window_size": window_size,
        }
