import cutlass
import cutlass.cute as cute


# params, grad, m, v are tensors on the GPU
@cute.jit
def solve(
    params: cute.Tensor,
    grad: cute.Tensor,
    m: cute.Tensor,
    v: cute.Tensor,
    N: cute.Int32,
    lr: cute.Float32,
    beta1: cute.Float32,
    beta2: cute.Float32,
    eps: cute.Float32,
    weight_decay: cute.Float32,
    t: cute.Int32,
):
    pass
