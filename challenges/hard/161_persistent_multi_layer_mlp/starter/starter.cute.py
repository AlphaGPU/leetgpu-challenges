import cutlass
import cutlass.cute as cute


# x, rms_w, W_gate, W_up, W_down, output are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    rms_w: cute.Tensor,
    W_gate: cute.Tensor,
    W_up: cute.Tensor,
    W_down: cute.Tensor,
    output: cute.Tensor,
    M: cute.Int32,
    d_model: cute.Int32,
    d_ffn: cute.Int32,
    L: cute.Int32,
):
    pass
