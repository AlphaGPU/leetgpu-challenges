import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase, OutTensor, RandTensor


class Challenge(ChallengeBase):
    name = "Gated Attention"
    # Same float32 tolerance as Multi-Head Attention (#12) and Causal Attention (#53). Outputs are
    # bounded by max |V| (gate in (0, 1)); on CPU the float32 reference stays within 15% of this
    # tolerance of a float64 implementation, worst on the large-score case. A missing causal mask,
    # a missing gate, W_g transposed or the gate applied to V before attention all miss by far more.
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
        assert d_model % h == 0

        # Head-specific gate logits for all heads at once: column block
        # [head * d_k : (head + 1) * d_k] of X @ W_g belongs to that head.
        gate = torch.sigmoid(torch.matmul(X, W_g))

        d_k = d_model // h
        causal = torch.ones(M, M, dtype=torch.bool, device=Q.device).tril()
        result = torch.empty_like(Q)
        for head in range(h):
            s = slice(head * d_k, (head + 1) * d_k)
            scores = torch.matmul(Q[:, s], K[:, s].t()) / (d_k**0.5)
            attn_weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
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
            "output": torch.full((3, 4), 123.0, device=self.device, dtype=dtype),
            "M": 3,
            "d_model": 4,
            "h": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests: List[Dict[str, Any]] = []

        # Every output buffer is pre-filled with 123.0, so a kernel that skips rows, heads or a
        # tile tail fails instead of passing on leftover zeros.
        #
        # 1: basic example (same as generate_example_test). W_g is not symmetric, so it also
        # catches using W_g transposed (X @ W_g.T); cases 2 and 7-12 catch it too.
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
        tests.append(
            {
                "Q": Q,
                "K": K,
                "V": V,
                "X": X,
                "W_g": W_g,
                "output": torch.full((1, 4), 123.0, device=self.device, dtype=dtype),
                "M": 1,
                "d_model": 4,
                "h": 2,
            }
        )

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
                "output": torch.full((4, 4), 123.0, device=self.device, dtype=dtype),
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
                "output": torch.full((4, 4), 123.0, device=self.device, dtype=dtype),
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
                "output": torch.full((4, 4), 123.0, device=self.device, dtype=dtype),
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
                "output": torch.full((4, 8), 123.0, device=self.device, dtype=dtype),
                "M": 4,
                "d_model": 8,
                "h": 2,
            }
        )

        # 7: mixed negative and positive values (deterministic literals). With per-element gates
        # that differ across tokens, applying the gate to V before attention (softmax(QK^T)(V*g))
        # gives a different result than gating the head output; cases 1 and 8-12 catch it too.
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
        tests.append(
            {
                "Q": Q,
                "K": K,
                "V": V,
                "X": X,
                "W_g": W_g,
                "output": torch.full((5, 8), 123.0, device=self.device, dtype=dtype),
                "M": 5,
                "d_model": 8,
                "h": 2,
            }
        )

        # 8: medium random case, 4 heads
        torch.manual_seed(42)
        tests.append(
            {
                "Q": self._rand((64, 64), -1.0, 1.0),
                "K": self._rand((64, 64), -1.0, 1.0),
                "V": self._rand((64, 64), -1.0, 1.0),
                "X": self._rand((64, 64), -1.0, 1.0),
                "W_g": self._rand((64, 64), -0.25, 0.25),
                "output": torch.full((64, 64), 123.0, device=self.device, dtype=dtype),
                "M": 64,
                "d_model": 64,
                "h": 4,
            }
        )

        # 9: head_dim 128 with a sequence length that is not a multiple of
        # typical tile sizes
        torch.manual_seed(123)
        tests.append(
            {
                "Q": self._rand((1031, 256), -1.0, 1.0),
                "K": self._rand((1031, 256), -1.0, 1.0),
                "V": self._rand((1031, 256), -1.0, 1.0),
                "X": self._rand((1031, 256), -1.0, 1.0),
                "W_g": self._rand((256, 256), -0.125, 0.125),
                "output": torch.full((1031, 256), 123.0, device=self.device, dtype=dtype),
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
                "output": torch.full((128, 64), 123.0, device=self.device, dtype=dtype),
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
                "output": torch.full((257, 128), 123.0, device=self.device, dtype=dtype),
                "M": 257,
                "d_model": 128,
                "h": 8,
            }
        )

        # 12: performance-test head dimension (d_model=512, h=8, d_k=64) with a
        # sequence length that is not a multiple of typical tile sizes
        torch.manual_seed(2025)
        tests.append(
            {
                "Q": self._rand((1031, 512), -1.0, 1.0),
                "K": self._rand((1031, 512), -1.0, 1.0),
                "V": self._rand((1031, 512), -1.0, 1.0),
                "X": self._rand((1031, 512), -1.0, 1.0),
                "W_g": self._rand((512, 512), -0.125, 0.125),
                "output": torch.full((1031, 512), 123.0, device=self.device, dtype=dtype),
                "M": 1031,
                "d_model": 512,
                "h": 8,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return {
            "Q": RandTensor((4096, 512), -1.0, 1.0),
            "K": RandTensor((4096, 512), -1.0, 1.0),
            "V": RandTensor((4096, 512), -1.0, 1.0),
            "X": RandTensor((4096, 512), -1.0, 1.0),
            "W_g": RandTensor((512, 512), -0.125, 0.125),
            "output": OutTensor((4096, 512)),
            "M": 4096,
            "d_model": 512,
            "h": 8,
        }
