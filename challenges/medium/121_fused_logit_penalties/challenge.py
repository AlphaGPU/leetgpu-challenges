import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused Logit Penalties"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        logits: torch.Tensor,
        prompt_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
        presence_penalty: torch.Tensor,
        frequency_penalty: torch.Tensor,
        repetition_penalty: torch.Tensor,
        output: torch.Tensor,
        B: int,
        V: int,
        P: int,
        G: int,
    ):
        assert logits.shape == (B, V)
        assert prompt_tokens.shape == (B, P)
        assert output_tokens.shape == (B, G)
        assert presence_penalty.shape == (B,)
        assert frequency_penalty.shape == (B,)
        assert repetition_penalty.shape == (B,)
        assert output.shape == (B, V)
        assert prompt_tokens.dtype == torch.int32
        assert output_tokens.dtype == torch.int32
        assert (
            logits.dtype
            == presence_penalty.dtype
            == frequency_penalty.dtype
            == repetition_penalty.dtype
            == output.dtype
        )

        dtype = logits.dtype

        prompt_idx = prompt_tokens.long()
        prompt_valid = prompt_idx >= 0
        prompt_idx = torch.where(prompt_idx >= 0, prompt_idx, torch.zeros_like(prompt_idx))
        prompt_counts = torch.zeros((B, V), dtype=dtype, device=logits.device)
        prompt_counts.scatter_add_(1, prompt_idx, prompt_valid.to(dtype))

        output_idx = output_tokens.long()
        output_valid = output_idx >= 0
        output_idx = torch.where(output_idx >= 0, output_idx, torch.zeros_like(output_idx))
        output_counts = torch.zeros((B, V), dtype=dtype, device=logits.device)
        output_counts.scatter_add_(1, output_idx, output_valid.to(dtype))

        seen = (prompt_counts + output_counts) > 0
        generated = output_counts > 0

        rep = repetition_penalty.reshape(B, 1)
        penalized = torch.where(logits > 0, logits / rep, logits * rep)
        result = torch.where(seen, penalized, logits)

        result = result - frequency_penalty.reshape(B, 1) * output_counts
        result = result - presence_penalty.reshape(B, 1) * generated.to(dtype)

        output.copy_(result)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "prompt_tokens": (ctypes.POINTER(ctypes.c_int), "in"),
            "output_tokens": (ctypes.POINTER(ctypes.c_int), "in"),
            "presence_penalty": (ctypes.POINTER(ctypes.c_float), "in"),
            "frequency_penalty": (ctypes.POINTER(ctypes.c_float), "in"),
            "repetition_penalty": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
            "P": (ctypes.c_int, "in"),
            "G": (ctypes.c_int, "in"),
        }

    def _build_case(
        self,
        logits: torch.Tensor,
        prompt_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
        presence: List[float],
        frequency: List[float],
        repetition: List[float],
    ) -> Dict[str, Any]:
        B, V = logits.shape
        return {
            "logits": logits,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "presence_penalty": torch.tensor(presence, device=self.device, dtype=torch.float32),
            "frequency_penalty": torch.tensor(frequency, device=self.device, dtype=torch.float32),
            "repetition_penalty": torch.tensor(repetition, device=self.device, dtype=torch.float32),
            "output": torch.empty(B, V, device=self.device, dtype=torch.float32),
            "B": B,
            "V": V,
            "P": prompt_tokens.shape[1],
            "G": output_tokens.shape[1],
        }

    def _pad_tail(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=self.device).reshape(1, -1)
        keep = positions < lengths.reshape(-1, 1)
        return torch.where(keep, tokens, torch.full_like(tokens, -1))

    def generate_example_test(self) -> Dict[str, Any]:
        logits = torch.tensor(
            [
                [1.0, -2.0, 3.0, 0.5, 0.0, -1.0],
                [2.0, 1.0, -1.0, 0.0, 4.0, -3.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        prompt_tokens = torch.tensor([[1, 3, -1], [0, 0, 2]], device=self.device, dtype=torch.int32)
        output_tokens = torch.tensor(
            [[2, 2, 5, -1], [4, -1, -1, -1]], device=self.device, dtype=torch.int32
        )
        return self._build_case(
            logits,
            prompt_tokens,
            output_tokens,
            presence=[0.5, 1.0],
            frequency=[0.25, 0.5],
            repetition=[2.0, 1.0],
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = []

        # single sequence, tiny vocab, the one generated token repeats the prompt
        tests.append(
            self._build_case(
                torch.tensor([[1.0, -1.0, 2.0, 0.0]], device=self.device, dtype=torch.float32),
                torch.tensor([[2]], device=self.device, dtype=torch.int32),
                torch.tensor([[2]], device=self.device, dtype=torch.int32),
                presence=[0.5],
                frequency=[0.75],
                repetition=[2.0],
            )
        )

        # all penalties disabled (repetition_penalty = 1) => output must equal logits
        tests.append(
            self._build_case(
                torch.tensor(
                    [[0.5, -0.5, 1.5], [-2.0, 3.0, -4.0]], device=self.device, dtype=torch.float32
                ),
                torch.tensor([[0, 1], [2, 2]], device=self.device, dtype=torch.int32),
                torch.tensor([[1, 1], [0, 2]], device=self.device, dtype=torch.int32),
                presence=[0.0, 0.0],
                frequency=[0.0, 0.0],
                repetition=[1.0, 1.0],
            )
        )

        # fully padded prompt and output: no token has been seen
        tests.append(
            self._build_case(
                torch.tensor([[-3.0, 2.0, 0.0, 7.0]], device=self.device, dtype=torch.float32),
                torch.full((1, 3), -1, device=self.device, dtype=torch.int32),
                torch.full((1, 2), -1, device=self.device, dtype=torch.int32),
                presence=[1.0],
                frequency=[1.0],
                repetition=[2.0],
            )
        )

        # heavy repeats of a single token, mixed signs, repetition penalty below 1
        tests.append(
            self._build_case(
                torch.tensor(
                    [
                        [-1.0, 4.0, -4.0, 0.0, 2.5, -2.5, 1.0, -1.5],
                        [3.0, 3.0, -3.0, -3.0, 0.5, 0.5, -0.5, -0.5],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[0, 0, 0, 7], [1, -1, -1, -1], [3, 4, 5, 6]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                torch.tensor(
                    [[1, 1, 1, 1, 1], [5, 5, 6, -1, -1], [-1, -1, -1, -1, 0]],
                    device=self.device,
                    dtype=torch.int32,
                ),
                presence=[0.25, 1.0, 0.5],
                frequency=[0.5, 0.125, 2.0],
                repetition=[0.5, 1.5, 2.0],
            )
        )

        # power-of-two vocab, all-zero logits
        torch.manual_seed(0)
        B, V, P, G = 2, 1024, 64, 32
        tests.append(
            self._build_case(
                torch.zeros(B, V, device=self.device, dtype=torch.float32),
                torch.randint(0, V, (B, P), device=self.device, dtype=torch.int32),
                torch.randint(0, 8, (B, G), device=self.device, dtype=torch.int32),
                presence=[0.5, 1.5],
                frequency=[0.25, 0.75],
                repetition=[1.2, 1.8],
            )
        )

        # non-power-of-two vocab with ragged (trailing padded) prompts
        torch.manual_seed(1)
        B, V, P, G = 8, 255, 64, 17
        lengths = torch.randint(0, P + 1, (B,), device=self.device)
        tests.append(
            self._build_case(
                torch.randn(B, V, device=self.device, dtype=torch.float32) * 3.0,
                self._pad_tail(
                    torch.randint(0, V, (B, P), device=self.device, dtype=torch.int32), lengths
                ),
                torch.randint(0, 32, (B, G), device=self.device, dtype=torch.int32),
                presence=[0.0, 0.5, 1.0, 1.5, 2.0, 0.25, 0.75, 1.25],
                frequency=[1.0, 0.0, 0.5, 0.25, 2.0, 0.125, 0.0, 1.5],
                repetition=[1.0, 1.1, 1.5, 2.0, 0.5, 1.25, 1.75, 1.05],
            )
        )

        # small non-power-of-two vocab, every row generates the same token repeatedly
        torch.manual_seed(2)
        B, V, P, G = 4, 100, 30, 30
        tests.append(
            self._build_case(
                torch.randn(B, V, device=self.device, dtype=torch.float32) * 2.0,
                torch.randint(0, 10, (B, P), device=self.device, dtype=torch.int32),
                torch.randint(0, 5, (B, G), device=self.device, dtype=torch.int32),
                presence=[1.0, 0.5, 0.0, 2.0],
                frequency=[0.5, 1.0, 0.25, 0.0],
                repetition=[1.5, 2.0, 1.0, 0.75],
            )
        )

        # wide batch, very short histories
        torch.manual_seed(3)
        B, V, P, G = 64, 1024, 4, 2
        presence = torch.rand(B).tolist()
        frequency = torch.rand(B).tolist()
        repetition = (1.0 + torch.rand(B)).tolist()
        tests.append(
            self._build_case(
                torch.randn(B, V, device=self.device, dtype=torch.float32) * 4.0,
                torch.randint(0, V, (B, P), device=self.device, dtype=torch.int32),
                torch.randint(0, V, (B, G), device=self.device, dtype=torch.int32),
                presence=presence,
                frequency=frequency,
                repetition=repetition,
            )
        )

        # realistic decode step
        torch.manual_seed(4)
        B, V, P, G = 16, 4096, 256, 128
        presence = torch.rand(B).tolist()
        frequency = torch.rand(B).tolist()
        repetition = (0.5 + 1.5 * torch.rand(B)).tolist()
        tests.append(
            self._build_case(
                torch.randn(B, V, device=self.device, dtype=torch.float32) * 2.5,
                torch.randint(0, V, (B, P), device=self.device, dtype=torch.int32),
                torch.randint(0, 512, (B, G), device=self.device, dtype=torch.int32),
                presence=presence,
                frequency=frequency,
                repetition=repetition,
            )
        )

        # realistic LLM vocab
        torch.manual_seed(5)
        B, V, P, G = 32, 32000, 512, 256
        lengths = torch.randint(1, P + 1, (B,), device=self.device)
        presence = torch.rand(B).tolist()
        frequency = torch.rand(B).tolist()
        repetition = (1.0 + torch.rand(B)).tolist()
        tests.append(
            self._build_case(
                torch.randn(B, V, device=self.device, dtype=torch.float32) * 2.0,
                self._pad_tail(
                    torch.randint(0, V, (B, P), device=self.device, dtype=torch.int32), lengths
                ),
                torch.randint(0, 2048, (B, G), device=self.device, dtype=torch.int32),
                presence=presence,
                frequency=frequency,
                repetition=repetition,
            )
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(42)
        B, V, P, G = 256, 128256, 1024, 512
        lengths = torch.randint(1, P + 1, (B,), device=self.device)
        return self._build_case(
            torch.randn(B, V, device=self.device, dtype=torch.float32) * 2.0,
            self._pad_tail(
                torch.randint(0, V, (B, P), device=self.device, dtype=torch.int32), lengths
            ),
            torch.randint(0, 8192, (B, G), device=self.device, dtype=torch.int32),
            presence=torch.rand(B).tolist(),
            frequency=torch.rand(B).tolist(),
            repetition=(1.0 + torch.rand(B)).tolist(),
        )
