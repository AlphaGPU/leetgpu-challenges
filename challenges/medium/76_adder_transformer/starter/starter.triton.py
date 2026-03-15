import torch
import triton
import triton.language as tl


# prompts, output, weights are tensors on the GPU
def solve(prompts: torch.Tensor, output: torch.Tensor, weights: torch.Tensor, batch_size: int):
    pass
