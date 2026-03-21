import cutlass
import cutlass.cute as cute


# x, W, A, B, output are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    W: cute.Tensor,
    A: cute.Tensor,
    B: cute.Tensor,
    output: cute.Tensor,
    batch: cute.Int32,
    d_in: cute.Int32,
    d_out: cute.Int32,
    rank: cute.Int32,
    lora_scale: cute.Float32,
):
    pass
