import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Nearest Neighbor",
            atol=0,
            rtol=0,
            num_gpus=1,
            access_tier="free"
        )

    def reference_impl(self, points: torch.Tensor, indices: torch.Tensor, N: int):
        """
        Reference implementation that finds the nearest neighbor for each point.
        For N three-dimensional points, fills indices[i] with the index j≠i 
        of the point closest to points[i].
        """
        assert points.dtype == torch.float32
        assert indices.dtype == torch.int32
        assert points.shape == (N * 3,)  # N points, each with 3 coordinates
        assert indices.shape == (N,)
        assert N >= 1

        # Reshape points to (N, 3) for easier processing
        pts = points.view(N, 3)
        
        # pts shape: (N, 3)
        # Expand to (N, 1, 3) and (1, N, 3) for broadcasting
        pts_expand1 = pts.unsqueeze(1)  # (N, 1, 3)
        pts_expand2 = pts.unsqueeze(0)  # (1, N, 3)
        
        # Compute all pairwise squared distances: (N, N)
        diff = pts_expand1 - pts_expand2  # (N, N, 3)
        dist_sq = torch.sum(diff * diff, dim=2)  # (N, N)
        
        # Mask diagonal (distance to self) with large value
        mask = torch.eye(N, device=points.device, dtype=torch.bool)
        dist_sq[mask] = float('inf')
        
        # Find nearest neighbor indices
        indices.copy_(torch.argmin(dist_sq, dim=1).int())

    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "points": ctypes.POINTER(ctypes.c_float),
            "indices": ctypes.POINTER(ctypes.c_int),
            "N": ctypes.c_int
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        N = 3
        
        # Example: points = [(0,0,0), (1,0,0), (5,5,5)]
        points_data = torch.tensor([0.0, 0.0, 0.0,  # point 0
                                   1.0, 0.0, 0.0,  # point 1
                                   5.0, 5.0, 5.0], # point 2
                                 device="cuda", dtype=dtype_float)
        indices_data = torch.full((N,), -1, device="cuda", dtype=dtype_int)

        return {
            "points": points_data,
            "indices": indices_data,
            "N": N
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        test_cases = []

        # Test case 1: Basic example from problem description
        test_cases.append({
            "points": torch.tensor([0.0, 0.0, 0.0,  # point 0
                                   1.0, 0.0, 0.0,  # point 1  
                                   5.0, 5.0, 5.0], # point 2
                                 device="cuda", dtype=dtype_float),
            "indices": torch.full((3,), -1, device="cuda", dtype=dtype_int),
            "N": 3
        })

        # Test case 2: Two points only
        test_cases.append({
            "points": torch.tensor([0.0, 0.0, 0.0,  # point 0
                                   3.0, 4.0, 0.0], # point 1
                                 device="cuda", dtype=dtype_float),
            "indices": torch.full((2,), -1, device="cuda", dtype=dtype_int),
            "N": 2
        })

        # Test case 3: Four points in a square
        test_cases.append({
            "points": torch.tensor([0.0, 0.0, 0.0,  # point 0
                                   1.0, 0.0, 0.0,  # point 1
                                   0.0, 1.0, 0.0,  # point 2
                                   1.0, 1.0, 0.0], # point 3
                                 device="cuda", dtype=dtype_float),
            "indices": torch.full((4,), -1, device="cuda", dtype=dtype_int),
            "N": 4
        })

        # Test case 4: Points with negative coordinates
        test_cases.append({
            "points": torch.tensor([-1.0, -1.0, -1.0,  # point 0
                                   1.0, 1.0, 1.0,     # point 1
                                   0.0, 0.0, 0.0,     # point 2
                                   2.0, 2.0, 2.0],    # point 3
                                 device="cuda", dtype=dtype_float),
            "indices": torch.full((4,), -1, device="cuda", dtype=dtype_int),
            "N": 4
        })

        # Test case 5: Points with clear unique nearest neighbors
        test_cases.append({
            "points": torch.tensor([0.0, 0.0, 0.0,   # point 0
                                   10.0, 0.0, 0.0,  # point 1
                                   1.0, 0.0, 0.0,   # point 2 (closest to 0)
                                   11.0, 0.0, 0.0,  # point 3 (closest to 1)
                                   5.0, 0.0, 0.0],  # point 4
                                 device="cuda", dtype=dtype_float),
            "indices": torch.full((5,), -1, device="cuda", dtype=dtype_int),
            "N": 5
        })

        # Test case 6: Medium random test with fixed seed for reproducibility
        torch.manual_seed(42)
        test_cases.append({
            "points": torch.empty((100, 3), device="cuda", dtype=dtype_float).uniform_(-100.0, 100.0).flatten(),
            "indices": torch.full((100,), -1, device="cuda", dtype=dtype_int),
            "N": 100
        })

        # Test case 7: Larger test with fixed seed
        torch.manual_seed(123)
        test_cases.append({
            "points": torch.empty((250, 3), device="cuda", dtype=dtype_float).uniform_(-1000.0, 1000.0).flatten(),
            "indices": torch.full((250,), -1, device="cuda", dtype=dtype_int),
            "N": 250
        })

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype_float = torch.float32
        dtype_int = torch.int32
        N = 10000
        
        return {
            "points": torch.empty((N, 3), device="cuda", dtype=dtype_float).uniform_(-1000.0, 1000.0).flatten(),
            "indices": torch.full((N,), -1, device="cuda", dtype=dtype_int),
            "N": N
        }