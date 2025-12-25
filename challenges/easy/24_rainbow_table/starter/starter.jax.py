import jax
import jax.numpy as jnp


def fnv1a_hash(x: jax.Array) -> jax.Array:
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    x_int = x.astype(jnp.int64)
    hash_val = jnp.full_like(x_int, OFFSET_BASIS, dtype=jnp.int64)

    for byte_pos in range(4):
        byte = (x_int >> (byte_pos * 8)) & 0xFF
        hash_val = hash_val ^ byte
        hash_val = (hash_val * FNV_PRIME) & 0xFFFFFFFF

    return hash_val


# input is a tensor on the GPU
def solve(input: jax.Array, N: int, R: int) -> jax.Array:
    # return output tensor directly
    pass
