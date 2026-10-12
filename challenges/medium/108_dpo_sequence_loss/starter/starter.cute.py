import cutlass
import cutlass.cute as cute


# chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are tensors on the GPU
@cute.jit
def solve(
    chosen_logps: cute.Tensor,
    rejected_logps: cute.Tensor,
    chosen_ref_logps: cute.Tensor,
    rejected_ref_logps: cute.Tensor,
    output: cute.Tensor,
    beta: cute.Float32,
    B: cute.Int32,
):
    pass
