import cutlass
import cutlass.cute as cute


# images, patch_weight, patch_bias, cls_token, pos_embed, output are tensors on the GPU
@cute.jit
def solve(
    images: cute.Tensor,
    patch_weight: cute.Tensor,
    patch_bias: cute.Tensor,
    cls_token: cute.Tensor,
    pos_embed: cute.Tensor,
    output: cute.Tensor,
    B: cute.Int32,
    C: cute.Int32,
    H: cute.Int32,
    W: cute.Int32,
    P: cute.Int32,
    D: cute.Int32,
):
    pass
