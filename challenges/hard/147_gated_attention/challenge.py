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
        X: torch.Tensor,
        W_g: torch.Tensor,
        output: torch.Tensor,
        M: int,
        d_model: int,
        h: int,
    ):
        assert Q.shape == (M, d_model)
        assert K.shape == (M, d_model)
        assert V.shape == (M, d_model)
        assert X.shape == (M, d_model)
        assert W_g.shape == (d_model, d_model)
        assert output.shape == (M, d_model)
        assert Q.dtype == K.dtype == V.dtype == output.dtype
        assert X.dtype == W_g.dtype == output.dtype

        # Head-specific gate logits for all heads at once: column block
        # [head * d_k : (head + 1) * d_k] of X @ W_g belongs to that head.
        gate = torch.sigmoid(torch.matmul(X, W_g))

        d_k = d_model // h
        mask = torch.triu(torch.full((M, M), float("-inf"), device=self.device), diagonal=1)
        result = torch.empty_like(Q)
        for head in range(h):
            s = slice(head * d_k, (head + 1) * d_k)
            scores = torch.matmul(Q[:, s], K[:, s].t()) / (d_k**0.5)
            attn_weights = torch.softmax(scores + mask, dim=-1)
            head_output = torch.matmul(attn_weights, V[:, s])
            result[:, s] = head_output * gate[:, s]
        output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "X": (ctypes.POINTER(ctypes.c_float), "in"),
            "W_g": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "d_model": (ctypes.c_int, "in"),
            "h": (ctypes.c_int, "in"),
        }

    def _rand(self, shape, low: float, high: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(low, high)

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        X = torch.tensor(
            [[1.0, 0.0, 2.0, 1.0], [0.0, 1.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0]],
            device=self.device,
            dtype=dtype,
        )
        W_g = torch.tensor(
            [
                [1.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        Q = torch.tensor(
            [[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 1.0, 2.0]],
            device=self.device,
            dtype=dtype,
        )
        K = torch.tensor(
            [[2.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 2.0], [0.0, 2.0, 1.0, 0.0]],
            device=self.device,
            dtype=dtype,
        )
        V = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "X": X,
            "W_g": W_g,
            "M": 3,
            "d_model": 4,
            "h": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests: List[Dict[str, Any]] = []

        # 1: basic example (same as generate_example_test)
        tests.append(self.generate_example_test())

        # 2: single token — attention is trivial, only the gate acts:
        # output = V * sigmoid(X @ W_g)
        X = torch.tensor([[1.0, -1.0, 2.0, 0.0]], device=self.device, dtype=dtype)
        W_g = torch.tensor(
            [
                [1.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, -1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 1.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        Q = torch.tensor([[0.5, 1.5, -1.0, 2.0]], device=self.device, dtype=dtype)
        K = torch.tensor([[1.0, 0.0, -0.5, 1.0]], device=self.device, dtype=dtype)
        V = torch.tensor([[3.0, -2.0, 1.0, 0.5]], device=self.device, dtype=dtype)
        tests.append({"Q": Q, "K": K, "V": V, "X": X, "W_g": W_g, "M": 1, "d_model": 4, "h": 2})

        # 3: zero gate weights — every gate is exactly 0.5, so the output is
        # half of plain multi-head causal attention
        torch.manual_seed(0)
        tests.append(
            {
                "Q": self._rand((4, 4), -1.0, 1.0),
                "K": self._rand((4, 4), -1.0, 1.0),
                "V": self._rand((4, 4), -1.0, 1.0),
                "X": self._rand((4, 4), -1.0, 1.0),
                "W_g": torch.zeros((4, 4), device=self.device, dtype=dtype),
                "M": 4,
                "d_model": 4,
                "h": 2,
            }
        )

        # 4: gate fully open — X @ W_g = +50 everywhere, so sigmoid = 1 and
        # the output is plain multi-head causal attention
        torch.manual_seed(1)
        tests.append(
            {
                "Q": self._rand((4, 4), -1.0, 1.0),
                "K": self._rand((4, 4), -1.0, 1.0),
                "V": self._rand((4, 4), -1.0, 1.0),
                "X": torch.ones((4, 4), device=self.device, dtype=dtype),
                "W_g": torch.full((4, 4), 12.5, device=self.device, dtype=dtype),
                "M": 4,
                "d_model": 4,
                "h": 2,
            }
        )

        # 5: gate fully closed — X @ W_g = -50 everywhere, so the output is
        # zero no matter what attention computes
        torch.manual_seed(2)
        tests.append(
            {
                "Q": self._rand((4, 4), -1.0, 1.0),
                "K": self._rand((4, 4), -1.0, 1.0),
                "V": self._rand((4, 4), -1.0, 1.0),
                "X": torch.ones((4, 4), device=self.device, dtype=dtype),
                "W_g": torch.full((4, 4), -12.5, device=self.device, dtype=dtype),
                "M": 4,
                "d_model": 4,
                "h": 2,
            }
        )

        # 6: all-zero inputs
        tests.append(
            {
                "Q": torch.zeros((4, 8), device=self.device, dtype=dtype),
                "K": torch.zeros((4, 8), device=self.device, dtype=dtype),
                "V": torch.zeros((4, 8), device=self.device, dtype=dtype),
                "X": torch.zeros((4, 8), device=self.device, dtype=dtype),
                "W_g": torch.zeros((8, 8), device=self.device, dtype=dtype),
                "M": 4,
                "d_model": 8,
                "h": 2,
            }
        )

        # 7: mixed negative and positive values (deterministic literals)
        X = torch.tensor(
            [
                [1.0, -1.0, 0.5, 2.0, -0.5, 1.0, 0.0, -2.0],
                [0.0, 2.0, -1.0, 1.0, 1.5, -1.0, 0.5, 0.0],
                [-1.0, 0.5, 2.0, 0.0, 1.0, 2.0, -1.5, 1.0],
                [2.0, 1.0, -0.5, -1.0, 0.0, 0.5, 1.0, -0.5],
                [0.5, 0.0, 1.0, -2.0, 2.0, -1.0, 0.0, 1.5],
            ],
            device=self.device,
            dtype=dtype,
        )
        W_g = torch.tensor(
            [
                [0.5, 0.0, -1.0, 0.5, 1.0, -0.5, 0.0, 1.0],
                [1.0, -0.5, 0.0, 1.0, -1.0, 0.5, 0.5, 0.0],
                [0.0, 1.0, 0.5, -0.5, 0.5, 1.0, -1.0, 0.5],
                [-0.5, 0.5, 1.0, 0.0, 0.0, -1.0, 0.5, 1.0],
                [1.0, 0.0, -0.5, 1.0, 0.5, 0.0, 1.0, -0.5],
                [0.5, 1.0, 0.0, -1.0, -0.5, 1.0, 0.0, 0.5],
                [-1.0, 0.5, 1.0, 0.5, 1.0, 0.0, -0.5, 0.0],
                [0.0, -1.0, 0.5, 1.0, 0.0, 0.5, 1.0, -1.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        Q = torch.tensor(
            [
                [0.5, -1.0, 2.0, 0.0, 1.0, 0.5, -0.5, 1.5],
                [1.0, 0.0, -0.5, 1.0, -2.0, 1.0, 0.5, 0.0],
                [-1.5, 1.0, 0.5, 2.0, 0.0, -1.0, 1.0, 0.5],
                [0.0, 2.0, 1.0, -1.0, 0.5, 0.0, -2.0, 1.0],
                [2.0, 0.5, 0.0, 0.5, 1.0, -0.5, 0.0, -1.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        K = torch.tensor(
            [
                [1.0, 0.5, -1.0, 1.5, 0.0, 2.0, 0.5, -0.5],
                [0.0, -1.0, 1.0, 0.0, 1.0, -0.5, 2.0, 1.0],
                [2.0, 1.0, 0.0, -0.5, -1.0, 0.5, 0.0, 2.0],
                [-0.5, 0.0, 2.0, 1.0, 0.5, 1.0, -1.0, 0.0],
                [1.5, -2.0, 0.5, 0.0, 2.0, 0.0, 1.0, 0.5],
            ],
            device=self.device,
            dtype=dtype,
        )
        V = torch.tensor(
            [
                [1.0, 2.0, -1.0, 0.5, 3.0, 0.0, -2.0, 1.0],
                [0.5, -1.5, 2.0, 1.0, 0.0, 2.5, 1.0, -0.5],
                [2.0, 0.0, 1.0, -1.0, -0.5, 1.5, 0.5, 2.0],
                [-1.0, 1.0, 0.5, 2.0, 1.0, -2.0, 3.0, 0.0],
                [0.0, 0.5, -2.0, 1.5, 2.0, 1.0, 0.0, -1.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        tests.append({"Q": Q, "K": K, "V": V, "X": X, "W_g": W_g, "M": 5, "d_model": 8, "h": 2})

        # 8: medium random case, 4 heads
        torch.manual_seed(42)
        tests.append(
            {
                "Q": self._rand((64, 64), -1.0, 1.0),
                "K": self._rand((64, 64), -1.0, 1.0),
                "V": self._rand((64, 64), -1.0, 1.0),
                "X": self._rand((64, 64), -1.0, 1.0),
                "W_g": self._rand((64, 64), -0.25, 0.25),
                "M": 64,
                "d_model": 64,
                "h": 4,
            }
        )

        # 9: head_dim 128 (same head_dim as the performance test) with a
        # sequence length that is not a multiple of typical tile sizes
        torch.manual_seed(123)
        tests.append(
            {
                "Q": self._rand((1031, 256), -1.0, 1.0),
                "K": self._rand((1031, 256), -1.0, 1.0),
                "V": self._rand((1031, 256), -1.0, 1.0),
                "X": self._rand((1031, 256), -1.0, 1.0),
                "W_g": self._rand((256, 256), -0.125, 0.125),
                "M": 1031,
                "d_model": 256,
                "h": 2,
            }
        )

        # 10: large attention scores — a softmax without max subtraction
        # overflows here
        torch.manual_seed(7)
        tests.append(
            {
                "Q": self._rand((128, 64), -8.0, 8.0),
                "K": self._rand((128, 64), -8.0, 8.0),
                "V": self._rand((128, 64), -1.0, 1.0),
                "X": self._rand((128, 64), -1.0, 1.0),
                "W_g": self._rand((64, 64), -0.25, 0.25),
                "M": 128,
                "d_model": 64,
                "h": 4,
            }
        )

        # 11: many heads, non-power-of-two sequence length
        torch.manual_seed(99)
        tests.append(
            {
                "Q": self._rand((257, 128), -2.0, 2.0),
                "K": self._rand((257, 128), -2.0, 2.0),
                "V": self._rand((257, 128), -2.0, 2.0),
                "X": self._rand((257, 128), -1.0, 1.0),
                "W_g": self._rand((128, 128), -0.25, 0.25),
                "M": 257,
                "d_model": 128,
                "h": 8,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return {
            "Q": RandTensor((4096, 512)),
            "K": RandTensor((4096, 512)),
            "V": RandTensor((4096, 512)),
            "X": RandTensor((4096, 512)),
            "W_g": RandTensor((512, 512)),
            "output": OutTensor((4096, 512)),
            "M": 4096,
            "d_model": 512,
            "h": 8,
        }

    def get_functional_test_count(self) -> int:
        return 11

    def get_performance_test_count(self) -> int:
        return 1
