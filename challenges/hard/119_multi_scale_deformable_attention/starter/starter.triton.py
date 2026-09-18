import torch
import triton
import triton.language as tl


# value, spatial_shapes, sampling_loc, attn_weight, output are tensors on the GPU
def solve(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_loc: torch.Tensor,
    attn_weight: torch.Tensor,
    output: torch.Tensor,
    num_queries: int,
    num_heads: int,
    head_dim: int,
    num_levels: int,
    num_points: int,
):
    pass
