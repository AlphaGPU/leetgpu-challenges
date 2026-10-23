import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Distillation KL Divergence"
    # Each output is sum_j p_j * (a_j - b_j) - LSE(a) + LSE(b). A float32 one-pass online
    # implementation differs from the reference by <= 3e-5 absolute on the functional cases
    # (cancellation between LSE terms of magnitude ~10-1000); 1e-3 leaves headroom for fast-math
    # exp. Reverse KL, missing temperature, an unnormalized student, or a stale online rescale
    # are off by > 1e-2 on the targeted cases.
    atol = 1e-03
    rtol = 1e-03
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        kl: torch.Tensor,
        N: int,
        V: int,
        temperature: float,
    ):
        assert teacher_logits.shape == (N, V)
        assert student_logits.shape == (N, V)
        assert kl.shape == (N,)
        assert teacher_logits.dtype == student_logits.dtype == kl.dtype == torch.float32

        log_p = torch.log_softmax(teacher_logits / temperature, dim=1)
        log_q = torch.log_softmax(student_logits / temperature, dim=1)
        kl.copy_(torch.sum(torch.exp(log_p) * (log_p - log_q), dim=1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "teacher_logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "student_logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "kl": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "V": (ctypes.c_int, "in"),
            "temperature": (ctypes.c_float, "in"),
        }

    def _case(self, teacher, student, temperature):
        N, V = teacher.shape
        return {
            "teacher_logits": teacher.contiguous(),
            "student_logits": student.contiguous(),
            # Pre-filled with garbage so solutions that accumulate into kl without
            # initializing it (e.g. atomicAdd onto an assumed-zero buffer) fail.
            "kl": torch.full((N,), 123.0, device=self.device, dtype=torch.float32),
            "N": N,
            "V": V,
            "temperature": temperature,
        }

    def _tensor(self, values):
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _random(self, N, V, temperature, std=3.0, corr=0.8, seed=0):
        # Student logits are correlated with the teacher's, as in real distillation.
        torch.manual_seed(seed)
        dev, f32 = self.device, torch.float32
        teacher = torch.randn(N, V, device=dev, dtype=f32) * std
        noise = torch.randn(N, V, device=dev, dtype=f32) * std
        student = corr * teacher + (1.0 - corr * corr) ** 0.5 * noise
        return self._case(teacher, student, temperature)

    def generate_example_test(self) -> Dict[str, Any]:
        teacher = self._tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        student = self._tensor([[3.0, 2.0, 1.0], [0.0, 0.0, 0.0]])
        return self._case(teacher, student, 1.0)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        tests = [self.generate_example_test()]

        # Same logits as the example at temperature 2. Hand-computed: p = softmax([.5, 1, 1.5])
        # = [0.18632, 0.30720, 0.50648], log p - log q = (a - b) = [-1, 0, 1] (equal LSEs since
        # b is a permutation of a), so kl[0] = 0.50648 - 0.18632 = 0.32016 and kl[1] = 0.
        # Catches ignoring the temperature (gives 1.1504 for row 0).
        ex2 = self.generate_example_test()
        ex2["temperature"] = 2.0
        tests.append(ex2)

        # V = 1: both distributions are the point mass, kl must be exactly 0.
        tests.append(self._case(self._tensor([[5.0]]), self._tensor([[-7.0]]), 1.0))

        # Zero teacher (uniform p) against a peaked student, plus all-zero rows (kl = 0).
        # Hand-computed row 0: KL(uniform || softmax([4,0,0,0])) = LSE(b) - mean(b) - log(V)
        # = log(e^4 + 3) - 1 - log(4) = 1.6672. Reverse KL here would be 1.1245.
        tests.append(
            self._case(
                self._tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
                self._tensor([[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
                1.0,
            )
        )

        # Peaked teacher vs. flat student: KL(p||q) and KL(q||p) differ by a large margin,
        # catching swapped teacher/student roles.
        t = torch.zeros(4, 64, device=self.device)
        t[:, 7] = 10.0
        s = torch.linspace(-0.5, 0.5, 64, device=self.device).repeat(4, 1)
        tests.append(self._case(t, s, 1.0))

        # Student = teacher + per-row constant: softmax is shift-invariant so kl = 0 exactly.
        # Catches an unnormalized student (dropping LSE(b)) or using the teacher's max/LSE
        # for the student's distribution.
        c = self._random(3, 100, 1.0, seed=1)
        shift = self._tensor([[20.0], [-20.0], [5.0]])
        tests.append(self._case(c["teacher_logits"], c["teacher_logits"] + shift, 1.0))

        # Large magnitudes with small temperature (scaled logits up to +-1000) and the student
        # shifted far above the teacher with maxima at different positions: exp() without
        # max-subtraction overflows, and reusing the teacher's max for the student overflows.
        torch.manual_seed(2)
        t = torch.rand(8, 255, device=self.device) * 600.0 - 300.0
        s = torch.rand(8, 255, device=self.device) * 300.0 + 200.0
        tests.append(self._case(t, s, 0.5))

        # Strictly increasing teacher and decreasing student: the running max of the teacher
        # changes at every chunk of every row, so a stale (un-rescaled) weighted accumulator in
        # a one-pass online algorithm is caught. V = 5000 is not a multiple of any block size.
        ramp = torch.linspace(-20.0, 20.0, 5000, device=self.device)
        tests.append(self._case(ramp.repeat(2, 1), (-0.5 * ramp).repeat(2, 1), 1.5))

        # All-negative logits.
        neg = self._random(16, 32, 1.0, seed=3)
        neg["teacher_logits"] = -neg["teacher_logits"].abs() - 1.0
        neg["student_logits"] = -neg["student_logits"].abs() - 1.0
        tests.append(neg)

        # Power-of-two shapes, temperature < 1 and > 1.
        tests.append(self._random(64, 1024, 0.5, seed=4))
        tests.append(self._random(16, 32768, 4.0, seed=5))

        # Non-power-of-two shapes (row tails).
        tests.append(self._random(30, 255, 1.0, seed=6))
        tests.append(self._random(100, 1000, 2.0, seed=7))

        # Realistic: Llama-2 vocabulary, many rows.
        tests.append(self._random(256, 32000, 1.0, seed=8))

        # Few very long rows (Llama-3 vocabulary): a row (513 KB) does not fit in shared memory,
        # catching solutions that stage a whole row on-chip.
        tests.append(self._random(4, 128256, 2.0, seed=9))

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        return self._random(2048, 128256, 2.0, seed=0)
