import cutlass
import cutlass.cute as cute


# teacher_logits, student_logits, kl are tensors on the GPU
@cute.jit
def solve(
    teacher_logits: cute.Tensor,
    student_logits: cute.Tensor,
    kl: cute.Tensor,
    N: cute.Int32,
    V: cute.Int32,
    temperature: cute.Float32,
):
    pass
