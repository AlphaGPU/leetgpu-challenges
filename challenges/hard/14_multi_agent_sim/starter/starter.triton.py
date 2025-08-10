import torch
import triton
import triton.language as tl

# agents, agents_next are tensors on the GPU
def solve(agents: torch.Tensor, agents_next: torch.Tensor, N: int):
    pass 