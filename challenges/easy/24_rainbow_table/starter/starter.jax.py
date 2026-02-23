import jax
import jax.numpy as jnp


def fnv1a_hash(x: jax.Array) -> jax.Array:
    FNV_PRIME = jnp.uint32(16777619)
    OFFSET_BASIS = jnp.uint32(2166136261)
    x_u32 = x.astype(jnp.uint32)
    hash_val = jnp.full_like(x_u32, OFFSET_BASIS, dtype=jnp.uint32)

    MASK_FF = jnp.uint32(0xFF)
    for byte_pos in range(4):
        byte = (x_u32 >> jnp.uint32(byte_pos * 8)) & MASK_FF
        hash_val = hash_val ^ byte
        hash_val = hash_val * FNV_PRIME

    return hash_val


# input is a tensor on the GPU
def solve(input: jax.Array, N: int, R: int) -> jax.Array:
    # return output tensor directly
    pass
