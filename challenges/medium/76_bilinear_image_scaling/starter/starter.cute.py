import cutlass
import cutlass.cute as cute


# image, output are tensors on the GPU
@cute.jit
def solve(
    image: cute.Tensor,
    output: cute.Tensor,
    H: cute.Int32,
    W: cute.Int32,
    H_out: cute.Int32,
    W_out: cute.Int32,
):
    pass
