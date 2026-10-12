import cutlass
import cutlass.cute as cute


# prompts, output, weights are tensors on the GPU
@cute.jit
def solve(
    prompts: cute.Tensor,
    output: cute.Tensor,
    weights: cute.Tensor,
    batch_size: cute.Int32,
):
    pass
