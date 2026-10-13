import cutlass
import cutlass.cute as cute


# x, topk_ids, topk_weights, w_gate, w_up, w_down, output are tensors on the GPU
@cute.jit
def solve(
    x: cute.Tensor,
    topk_ids: cute.Tensor,
    topk_weights: cute.Tensor,
    w_gate: cute.Tensor,
    w_up: cute.Tensor,
    w_down: cute.Tensor,
    output: cute.Tensor,
    T: cute.Int32,
    H: cute.Int32,
    I: cute.Int32,
    E: cute.Int32,
    K: cute.Int32,
):
    pass
