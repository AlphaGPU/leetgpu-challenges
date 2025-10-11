import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Categorical Cross Entropy Loss",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int):
        assert logits.dtype == torch.float32
        assert true_labels.dtype == torch.int32
        assert loss.dtype == torch.float32
        assert logits.shape == (N, C)
        assert true_labels.shape == (N,)
        assert loss.shape == (1,)
        assert N > 0 and C > 0
        total_loss = 0.0
        for i in range(N):
            log_probs = logits[i]
            true_label = true_labels[i].item()
            assert 0 <= true_label < C
            max_logit = torch.max(log_probs)
            log_sum_exp = max_logit + torch.log(torch.sum(torch.exp(log_probs - max_logit)))
            loss_i = log_sum_exp - log_probs[true_label]
            total_loss += loss_i.item()
        loss[0] = total_loss / N

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "true_labels": (ctypes.POINTER(ctypes.c_int), "in"),
            "loss": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "C": (ctypes.c_int, "in")
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype_logits = torch.float32
        dtype_labels = torch.int32
        logits = torch.tensor([[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]], device="cuda", dtype=dtype_logits)
        true_labels = torch.tensor([1, 1], device="cuda", dtype=dtype_labels)
        loss = torch.zeros(1, device="cuda", dtype=dtype_logits)
        return {
            "logits": logits,
            "true_labels": true_labels,
            "loss": loss,
            "N": 2,
            "C": 3
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype_logits = torch.float32
        dtype_labels = torch.int32
        tests = []
        # basic_example
        tests.append({
            "logits": torch.tensor([[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]], device="cuda", dtype=dtype_logits),
            "true_labels": torch.tensor([1, 1], device="cuda", dtype=dtype_labels),
            "loss": torch.zeros(1, device="cuda", dtype=dtype_logits),
            "N": 2,
            "C": 3
        })
        # example_2
        tests.append({
            "logits": torch.tensor([[-0.5, 1.5, 0.0, 1.0], [2.0, -1.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]], device="cuda", dtype=dtype_logits),
            "true_labels": torch.tensor([3, 0, 1], device="cuda", dtype=dtype_labels),
            "loss": torch.zeros(1, device="cuda", dtype=dtype_logits),
            "N": 3,
            "C": 4
        })
        # single_sample
        tests.append({
            "logits": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], device="cuda", dtype=dtype_logits),
            "true_labels": torch.tensor([4], device="cuda", dtype=dtype_labels),
            "loss": torch.zeros(1, device="cuda", dtype=dtype_logits),
            "N": 1,
            "C": 5
        })
        # uniform_logits_correct_label
        tests.append({
            "logits": torch.tensor([[1.0]*5, [1.0]*5], device="cuda", dtype=dtype_logits),
            "true_labels": torch.tensor([0, 0], device="cuda", dtype=dtype_labels),
            "loss": torch.zeros(1, device="cuda", dtype=dtype_logits),
            "N": 2,
            "C": 5
        })
        # high_confidence_correct
        tests.append({
            "logits": torch.tensor([[-5.0, -5.0, 10.0, -5.0], [10.0, -5.0, -5.0, -5.0]], device="cuda", dtype=dtype_logits),
            "true_labels": torch.tensor([2, 0], device="cuda", dtype=dtype_labels),
            "loss": torch.zeros(1, device="cuda", dtype=dtype_logits),
            "N": 2,
            "C": 4
        })
        # high_confidence_incorrect
        tests.append({
            "logits": torch.tensor([[10.0, -5.0, -5.0], [-5.0, 10.0, -5.0]], device="cuda", dtype=dtype_logits),
            "true_labels": torch.tensor([1, 2], device="cuda", dtype=dtype_labels),
            "loss": torch.zeros(1, device="cuda", dtype=dtype_logits),
            "N": 2,
            "C": 3
        })
        # larger_batch_random
        tests.append({
            "logits": torch.empty(100, 5, device="cuda", dtype=dtype_logits).uniform_(-5.0, 5.0),
            "true_labels": torch.randint(0, 5, (100,), device="cuda", dtype=dtype_labels),
            "loss": torch.zeros(1, device="cuda", dtype=dtype_logits),
            "N": 100,
            "C": 5
        })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype_logits = torch.float32
        dtype_labels = torch.int32
        logits = torch.empty(10000, 1000, device="cuda", dtype=dtype_logits).uniform_(-10.0, 10.0)
        true_labels = torch.randint(0, 1000, (10000,), device="cuda", dtype=dtype_labels)
        loss = torch.zeros(1, device="cuda", dtype=dtype_logits)
        return {
            "logits": logits,
            "true_labels": true_labels,
            "loss": loss,
            "N": 10000,
            "C": 1000
        } 
