from memory   import UnsafePointer
from gpu.kernel import launch_kernel

# points  : pointer to N*3 Float32  (device)
# indices : pointer to N   Int32    (device, output)
# N       : Int32
@export
def solve(points:  UnsafePointer[Float32],
          indices: UnsafePointer[Int32],
          N:       Int32):
    """
    Mojo starter stub – launch your GPU kernel here.
    """
    pass   # TODO
