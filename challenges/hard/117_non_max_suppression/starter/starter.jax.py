import jax
import jax.numpy as jnp


# boxes, scores are tensors on device
@jax.jit
def solve(boxes: jax.Array, scores: jax.Array, N: int, iou_threshold: float) -> jax.Array:
    # return output tensor directly
    pass
