import cutlass
import cutlass.cute as cute


# image_embeds, text_embeds, loss are tensors on the GPU
@cute.jit
def solve(
    image_embeds: cute.Tensor,
    text_embeds: cute.Tensor,
    loss: cute.Tensor,
    N: cute.Int32,
    D: cute.Int32,
    logit_scale: cute.Float32,
):
    pass
