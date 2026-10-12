import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Token Embedding Layer"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        position_embeddings: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        output: torch.Tensor,
        B: int,
        T: int,
        V: int,
        P: int,
        D: int,
        eps: float,
    ):
        assert token_ids.shape == (B, T)
        assert position_ids.shape == (T,)
        assert token_embeddings.shape == (V, D)
        assert position_embeddings.shape == (P, D)
        assert gamma.shape == (D,)
        assert beta.shape == (D,)
        assert output.shape == (B, T, D)
        assert token_ids.dtype == position_ids.dtype == torch.int32
        assert (
            token_embeddings.dtype
            == position_embeddings.dtype
            == gamma.dtype
            == beta.dtype
            == output.dtype
        )

        tok = token_embeddings[token_ids.long()]
        pos = position_embeddings[position_ids.long()]
        summed = tok + pos.unsqueeze(0)

        mean = summed.mean(dim=-1, keepdim=True)
        var = summed.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (summed - mean) * torch.rsqrt(var + eps)
        output.copy_(normalized * gamma + beta)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "token_ids": (ctypes.POINTER(ctypes.c_int), "in"),
            "position_ids": (ctypes.POINTER(ctypes.c_int), "in"),
            "token_embeddings": (ctypes.POINTER(ctypes.c_float), "in"),
            "position_embeddings": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "B": (ctypes.c_int, "in"),
            "T": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
            "P": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def _build_case(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        position_embeddings: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        B: int,
        T: int,
        V: int,
        P: int,
        D: int,
        eps: float,
    ) -> Dict[str, Any]:
        return {
            "token_ids": token_ids,
            "position_ids": position_ids,
            "token_embeddings": token_embeddings,
            "position_embeddings": position_embeddings,
            "gamma": gamma,
            "beta": beta,
            "output": torch.empty((B, T, D), device=self.device, dtype=torch.float32),
            "B": B,
            "T": T,
            "V": V,
            "P": P,
            "D": D,
            "eps": eps,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        idtype = torch.int32
        B, T, V, P, D = 1, 2, 3, 2, 4
        token_ids = torch.tensor([[2, 0]], device=self.device, dtype=idtype)
        position_ids = torch.tensor([0, 1], device=self.device, dtype=idtype)
        token_embeddings = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 0.0, -1.0],
                [2.0, 0.0, -2.0, 0.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        position_embeddings = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, -1.0, 0.0],
            ],
            device=self.device,
            dtype=dtype,
        )
        gamma = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype)
        beta = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype)
        return self._build_case(
            token_ids,
            position_ids,
            token_embeddings,
            position_embeddings,
            gamma,
            beta,
            B,
            T,
            V,
            P,
            D,
            1e-5,
        )

    def _random_case(
        self,
        B: int,
        T: int,
        V: int,
        P: int,
        D: int,
        eps: float = 1e-5,
        emb_range: tuple = (-1.0, 1.0),
        gamma_range: tuple = (0.5, 1.5),
        beta_range: tuple = (-0.5, 0.5),
    ) -> Dict[str, Any]:
        dtype = torch.float32
        idtype = torch.int32
        token_ids = torch.randint(0, V, (B, T), device=self.device, dtype=idtype)
        position_ids = torch.randint(0, P, (T,), device=self.device, dtype=idtype)
        token_embeddings = torch.empty((V, D), device=self.device, dtype=dtype).uniform_(
            emb_range[0], emb_range[1]
        )
        position_embeddings = torch.empty((P, D), device=self.device, dtype=dtype).uniform_(
            emb_range[0], emb_range[1]
        )
        gamma = torch.empty((D,), device=self.device, dtype=dtype).uniform_(
            gamma_range[0], gamma_range[1]
        )
        beta = torch.empty((D,), device=self.device, dtype=dtype).uniform_(
            beta_range[0], beta_range[1]
        )
        return self._build_case(
            token_ids,
            position_ids,
            token_embeddings,
            position_embeddings,
            gamma,
            beta,
            B,
            T,
            V,
            P,
            D,
            eps,
        )

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        idtype = torch.int32
        tests: List[Dict[str, Any]] = []

        # tiny single-token case
        tests.append(
            self._build_case(
                torch.tensor([[0]], device=self.device, dtype=idtype),
                torch.tensor([0], device=self.device, dtype=idtype),
                torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device, dtype=dtype),
                torch.tensor([[0.5, -0.5, 1.0, -1.0]], device=self.device, dtype=dtype),
                torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype),
                torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype),
                1,
                1,
                1,
                1,
                4,
                1e-5,
            )
        )

        # all-zero embeddings (degenerate variance handled by eps)
        B, T, V, P, D = 2, 3, 4, 3, 8
        tests.append(
            self._build_case(
                torch.randint(0, V, (B, T), device=self.device, dtype=idtype),
                torch.randint(0, P, (T,), device=self.device, dtype=idtype),
                torch.zeros((V, D), device=self.device, dtype=dtype),
                torch.zeros((P, D), device=self.device, dtype=dtype),
                torch.ones((D,), device=self.device, dtype=dtype),
                torch.zeros((D,), device=self.device, dtype=dtype),
                B,
                T,
                V,
                P,
                D,
                1e-5,
            )
        )

        # negative embeddings + non-trivial gamma/beta
        B, T, V, P, D = 2, 4, 5, 4, 8
        tests.append(
            self._build_case(
                torch.randint(0, V, (B, T), device=self.device, dtype=idtype),
                torch.tensor([0, 1, 2, 3], device=self.device, dtype=idtype),
                torch.empty((V, D), device=self.device, dtype=dtype).uniform_(-3.0, -0.5),
                torch.empty((P, D), device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty((D,), device=self.device, dtype=dtype).uniform_(0.5, 2.0),
                torch.empty((D,), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                B,
                T,
                V,
                P,
                D,
                1e-5,
            )
        )

        # power-of-two dims, repeated token ids (heavy gather collisions)
        B, T, V, P, D = 4, 16, 8, 16, 32
        token_ids = torch.zeros((B, T), device=self.device, dtype=idtype)
        for b in range(B):
            token_ids[b] = torch.tensor([b % V] * T, device=self.device, dtype=idtype)
        tests.append(
            self._build_case(
                token_ids,
                torch.arange(T, device=self.device, dtype=idtype) % P,
                torch.empty((V, D), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                torch.empty((P, D), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                torch.empty((D,), device=self.device, dtype=dtype).uniform_(0.5, 1.5),
                torch.empty((D,), device=self.device, dtype=dtype).uniform_(-0.5, 0.5),
                B,
                T,
                V,
                P,
                D,
                1e-5,
            )
        )

        # power-of-two medium
        tests.append(self._random_case(B=8, T=64, V=256, P=128, D=64))

        # non-power-of-two
        tests.append(self._random_case(B=3, T=17, V=100, P=33, D=48))

        # larger non-power-of-two
        tests.append(self._random_case(B=5, T=100, V=1000, P=255, D=192))

        # realistic small transformer-like
        tests.append(self._random_case(B=4, T=128, V=2048, P=512, D=256))

        # realistic larger
        tests.append(self._random_case(B=8, T=256, V=5000, P=512, D=384, emb_range=(-0.3, 0.3)))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._random_case(
            B=32,
            T=512,
            V=30000,
            P=2048,
            D=768,
            eps=1e-5,
            emb_range=(-0.3, 0.3),
            gamma_range=(0.8, 1.2),
            beta_range=(-0.1, 0.1),
        )
