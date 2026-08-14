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
    N: cute.Int32,
    S: cute.Int32,
    Q: cute.Int32,
    H: cute.Int32,
    D: cute.Int32,
    L: cute.Int32,
    P: cute.Int32,
):
    pass
