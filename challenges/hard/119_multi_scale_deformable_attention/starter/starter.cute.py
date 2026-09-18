import cutlass
import cutlass.cute as cute


# value, spatial_shapes, sampling_loc, attn_weight, output are tensors on the GPU
@cute.jit
def solve(
    value: cute.Tensor,
    spatial_shapes: cute.Tensor,
    sampling_loc: cute.Tensor,
    attn_weight: cute.Tensor,
    output: cute.Tensor,
    num_queries: cute.Int32,
    num_heads: cute.Int32,
    head_dim: cute.Int32,
    num_levels: cute.Int32,
    num_points: cute.Int32,
):
    pass
