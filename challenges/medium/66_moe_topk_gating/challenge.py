import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="MoE Top-K Gating", atol=1e-05, rtol=1e-05, num_gpus=1, access_tier="free"
        )

    def reference_impl(
        self,
        logits: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        M: int,
        E: int,
        k: int,
    ):
        """
        Computes the Top-K gating for Mixture of Experts.

        For each row in logits, select the k highest values, apply softmax to them,
        and return the weights and indices.
        """
        assert logits.shape == (M, E)
        assert topk_weights.shape == (M, k)
        assert topk_indices.shape == (M, k)
        assert logits.is_cuda and topk_weights.is_cuda and topk_indices.is_cuda
        assert topk_indices.dtype == torch.int32

        # 1. TopK Selection
        # logits: (M, E) -> vals: (M, k), indices: (M, k)
        vals, indices = torch.topk(logits, k, dim=-1)

        # 2. Softmax on the top k values
        weights = torch.softmax(vals, dim=-1)

        # 3. Write output
        topk_weights.copy_(weights)
        topk_indices.copy_(indices.to(torch.int32))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "topk_weights": (ctypes.POINTER(ctypes.c_float), "out"),
            "topk_indices": (ctypes.POINTER(ctypes.c_int), "out"),
            "M": (ctypes.c_int, "in"),
            "E": (ctypes.c_int, "in"),
            "k": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        M = 2
        E = 4
        k = 2

        # Example from problem description
        logits_data = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], device="cuda", dtype=dtype_float
        )
        topk_weights_data = torch.zeros((M, k), device="cuda", dtype=dtype_float)
        topk_indices_data = torch.zeros((M, k), device="cuda", dtype=dtype_int)

        return {
            "logits": logits_data,
            "topk_weights": topk_weights_data,
            "topk_indices": topk_indices_data,
            "M": M,
            "E": E,
            "k": k,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        test_cases = []

        # Test case 1: Basic example from problem description
        test_cases.append(
            {
                "logits": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], device="cuda", dtype=dtype_float
                ),
                "topk_weights": torch.zeros((2, 2), device="cuda", dtype=dtype_float),
                "topk_indices": torch.zeros((2, 2), device="cuda", dtype=dtype_int),
                "M": 2,
                "E": 4,
                "k": 2,
            }
        )

        # Test case 2: k=1 (single expert per token)
        test_cases.append(
            {
                "logits": torch.tensor(
                    [[5.0, 1.0, 3.0], [2.0, 8.0, 4.0], [6.0, 2.0, 9.0]],
                    device="cuda",
                    dtype=dtype_float,
                ),
                "topk_weights": torch.zeros((3, 1), device="cuda", dtype=dtype_float),
                "topk_indices": torch.zeros((3, 1), device="cuda", dtype=dtype_int),
                "M": 3,
                "E": 3,
                "k": 1,
            }
        )

        # Test case 3: k=E (all experts)
        test_cases.append(
            {
                "logits": torch.tensor(
                    [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], device="cuda", dtype=dtype_float
                ),
                "topk_weights": torch.zeros((2, 3), device="cuda", dtype=dtype_float),
                "topk_indices": torch.zeros((2, 3), device="cuda", dtype=dtype_int),
                "M": 2,
                "E": 3,
                "k": 3,
            }
        )

        # Test case 4: Typical MoE configuration (M=4, E=8, k=2)
        torch.manual_seed(42)
        test_cases.append(
            {
                "logits": torch.randn((4, 8), device="cuda", dtype=dtype_float),
                "topk_weights": torch.zeros((4, 2), device="cuda", dtype=dtype_float),
                "topk_indices": torch.zeros((4, 2), device="cuda", dtype=dtype_int),
                "M": 4,
                "E": 8,
                "k": 2,
            }
        )

        # Test case 5: Larger E with small k (M=8, E=64, k=2)
        torch.manual_seed(123)
        test_cases.append(
            {
                "logits": torch.randn((8, 64), device="cuda", dtype=dtype_float),
                "topk_weights": torch.zeros((8, 2), device="cuda", dtype=dtype_float),
                "topk_indices": torch.zeros((8, 2), device="cuda", dtype=dtype_int),
                "M": 8,
                "E": 64,
                "k": 2,
            }
        )

        # Test case 6: Test with negative logits
        test_cases.append(
            {
                "logits": torch.tensor(
                    [[-1.0, -2.0, -3.0, -4.0], [-4.0, -1.0, -2.0, -3.0]],
                    device="cuda",
                    dtype=dtype_float,
                ),
                "topk_weights": torch.zeros((2, 2), device="cuda", dtype=dtype_float),
                "topk_indices": torch.zeros((2, 2), device="cuda", dtype=dtype_int),
                "M": 2,
                "E": 4,
                "k": 2,
            }
        )

        # Test case 7: Medium size test (M=100, E=16, k=4)
        torch.manual_seed(456)
        test_cases.append(
            {
                "logits": torch.randn((100, 16), device="cuda", dtype=dtype_float),
                "topk_weights": torch.zeros((100, 4), device="cuda", dtype=dtype_float),
                "topk_indices": torch.zeros((100, 4), device="cuda", dtype=dtype_int),
                "M": 100,
                "E": 16,
                "k": 4,
            }
        )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        M = 1024
        E = 64
        k = 2

        torch.manual_seed(789)
        return {
            "logits": torch.randn((M, E), device="cuda", dtype=dtype_float),
            "topk_weights": torch.zeros((M, k), device="cuda", dtype=dtype_float),
            "topk_indices": torch.zeros((M, k), device="cuda", dtype=dtype_int),
            "M": M,
            "E": E,
            "k": k,
        }
