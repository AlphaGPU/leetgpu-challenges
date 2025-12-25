import cutlass
import cutlass.cute as cute

@cute.jit
def solve(logits: cute.Tensor, p: cute.Tensor, seed: cute.Tensor,
         sampled_token: cute.Tensor, vocab_size: cute.Int32):
    pass
