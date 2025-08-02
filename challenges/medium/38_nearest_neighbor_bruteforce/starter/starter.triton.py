import triton
import triton.language as tl

# points_ptr  – raw device pointer to N*3 float32
# indices_ptr – raw device pointer to N   int32  (output)
def solve(points_ptr: int, indices_ptr: int, N: int):
    """
    Triton starter stub.  Wrap the raw pointers and launch a custom kernel.
    """
    pass  # TODO
