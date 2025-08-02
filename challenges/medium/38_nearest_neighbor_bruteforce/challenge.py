import ctypes
from typing import Any, Dict
import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    """
    Brute-Force Nearest-Neighbor (functionality-only version).

    The runner passes three device pointers:

        points   —  float*  (N * 3 values, row-major)
        indices  —  int32*  (N values, to be filled in-place)
        N        —  int     (number of points)

    Your kernel must write, for every i,
        indices[i] = argmin_{j ≠ i} ||points[i] − points[j]||²
    """

    def __init__(self):
        super().__init__(
            name="Brute-Force Nearest Neighbor",
            atol=0.0,
            rtol=0.0,
            num_gpus=1,
            access_tier="free",
        )

    # ------------------------------------------------------------------ #
    # No reference implementation — correctness is verified externally.  #
    # ------------------------------------------------------------------ #

    def get_solve_signature(self) -> Dict[str, Any]:
        """C-ABI types expected by the harness."""
        return {
            "points":  ctypes.POINTER(ctypes.c_float),   # float32 [N*3]
            "indices": ctypes.POINTER(ctypes.c_int32),   # int32   [N]
            "N":       ctypes.c_int,
        }

    # --- optional small example shown on the challenge page ------------
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        pts = torch.tensor(
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [5.0, 5.0, 5.0]],
            device="cuda",
            dtype=dtype,
        )
        idx = torch.full((3,), -1, device="cuda", dtype=torch.int32)
        return {"points": pts, "indices": idx, "N": 3}

    # --- leave functional / performance tests to the evaluation server --
    def generate_functional_test(self):
        return []

    def generate_performance_test(self):
        return {}
