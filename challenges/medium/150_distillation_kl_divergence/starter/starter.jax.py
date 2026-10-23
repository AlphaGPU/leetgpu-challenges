import jax
import jax.numpy as jnp


# teacher_logits, student_logits are tensors on device
@jax.jit
def solve(
    teacher_logits: jax.Array, student_logits: jax.Array, N: int, V: int, temperature: float
) -> jax.Array:
    # return output tensor directly
    pass
