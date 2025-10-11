import ctypes
from typing import Any, List, Dict
import torch
from core.challenge_base import ChallengeBase

class Challenge(ChallengeBase):
    def __init__(self):
        super().__init__(
            name="Multi-Agent Simulation",
            atol=1e-05,
            rtol=1e-05,
            num_gpus=1,
            access_tier="free"
        )
        
    def reference_impl(self, agents: torch.Tensor, agents_next: torch.Tensor, N: int):
        assert agents.shape == (4 * N,)
        assert agents_next.shape == (4 * N,)
        assert agents.dtype == agents_next.dtype
        assert agents.device == agents_next.device
        r = 5.0
        r2 = r * r
        alpha = 0.05
        agents_reshaped = agents.view(N, 4)
        agents_next_reshaped = agents_next.view(N, 4)
        positions = agents_reshaped[:, :2]
        velocities = agents_reshaped[:, 2:]
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        dist_sq.fill_diagonal_(r2 + 1)
        neighbor_mask = dist_sq < r2
        sum_velocities = neighbor_mask.float() @ velocities
        neighbor_counts = neighbor_mask.sum(dim=1, keepdim=True)
        avg_velocities = torch.empty_like(velocities)
        nonzero_mask = (neighbor_counts[:, 0] > 0)
        avg_velocities[nonzero_mask] = sum_velocities[nonzero_mask] / neighbor_counts[nonzero_mask]
        avg_velocities[~nonzero_mask] = velocities[~nonzero_mask]
        new_velocities = velocities + alpha * (avg_velocities - velocities)
        new_positions = positions + new_velocities
        agents_next_reshaped[:] = torch.cat([new_positions, new_velocities], dim=1)
        agents_next.copy_(agents_next_reshaped.view(-1))
    
    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "agents": (ctypes.POINTER(ctypes.c_float), "in"),
            "agents_next": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in")
        }
    
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 2
        agents = torch.tensor([0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 1.0], device="cuda", dtype=dtype)
        agents_next = torch.empty(4 * N, device="cuda", dtype=dtype)
        return {
            "agents": agents,
            "agents_next": agents_next,
            "N": N
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []
        # basic_example
        agents = torch.tensor([0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 0.0, -1.0], device="cuda", dtype=dtype)
        agents_next = torch.empty(8, device="cuda", dtype=dtype)
        test_cases.append({
            "agents": agents,
            "agents_next": agents_next,
            "N": 2
        })
        # single_agent
        agents = torch.tensor([10.0, 15.0, 1.0, -1.0], device="cuda", dtype=dtype)
        agents_next = torch.empty(4, device="cuda", dtype=dtype)
        test_cases.append({
            "agents": agents,
            "agents_next": agents_next,
            "N": 1
        })
        # two_agents_interacting
        agents = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], device="cuda", dtype=dtype)
        agents_next = torch.empty(8, device="cuda", dtype=dtype)
        test_cases.append({
            "agents": agents,
            "agents_next": agents_next,
            "N": 2
        })
        # four_agents
        agents = torch.tensor([
            0.0, 0.0, 1.0, 0.0,
            2.0, 2.0, 0.0, 1.0,
            4.0, 4.0, -1.0, 0.0,
            6.0, 6.0, 0.0, -1.0
        ], device="cuda", dtype=dtype)
        agents_next = torch.empty(16, device="cuda", dtype=dtype)
        test_cases.append({
            "agents": agents,
            "agents_next": agents_next,
            "N": 4
        })
        # boundary_distance
        agents = torch.tensor([0.0, 0.0, 1.0, 1.0, 3.0, 4.0, -1.0, -1.0], device="cuda", dtype=dtype)
        agents_next = torch.empty(8, device="cuda", dtype=dtype)
        test_cases.append({
            "agents": agents,
            "agents_next": agents_next,
            "N": 2
        })
        # medium_simulation (random)
        agents = torch.empty(4096, device="cuda", dtype=dtype).uniform_(-100.0, 100.0)
        agents_next = torch.empty(4096, device="cuda", dtype=dtype)
        test_cases.append({
            "agents": agents,
            "agents_next": agents_next,
            "N": 1024
        })
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        agents = torch.empty(40000, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0)
        agents_next = torch.empty(40000, device="cuda", dtype=dtype)
        return {
            "agents": agents,
            "agents_next": agents_next,
            "N": 10000
        } 