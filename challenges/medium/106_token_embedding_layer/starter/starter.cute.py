import cutlass
import cutlass.cute as cute


# token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output
# are tensors on the GPU
@cute.jit
def solve(
    token_ids: cute.Tensor,
    position_ids: cute.Tensor,
    token_embeddings: cute.Tensor,
    position_embeddings: cute.Tensor,
    gamma: cute.Tensor,
    beta: cute.Tensor,
    output: cute.Tensor,
    B: cute.Int32,
    T: cute.Int32,
    V: cute.Int32,
    P: cute.Int32,
    D: cute.Int32,
    eps: cute.Float32,
):
    pass
