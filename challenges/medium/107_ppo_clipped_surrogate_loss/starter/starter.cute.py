import cutlass
import cutlass.cute as cute


# advantages, log_pi, log_pi_old, output are tensors on the GPU
@cute.jit
def solve(
    advantages: cute.Tensor,
    log_pi: cute.Tensor,
    log_pi_old: cute.Tensor,
    output: cute.Tensor,
    clip_eps: cute.Float32,
    B: cute.Int32,
    S: cute.Int32,
):
    pass
