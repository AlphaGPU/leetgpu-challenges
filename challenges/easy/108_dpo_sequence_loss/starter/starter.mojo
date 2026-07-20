from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.math import ceildiv


def dpo_loss_kernel(
    chosen_logps: UnsafePointer[Float32, MutExternalOrigin],
    rejected_logps: UnsafePointer[Float32, MutExternalOrigin],
    chosen_ref_logps: UnsafePointer[Float32, MutExternalOrigin],
    rejected_ref_logps: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    beta: Float32,
    B: Int32,
):
    pass


# chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(
    chosen_logps: UnsafePointer[Float32, MutExternalOrigin],
    rejected_logps: UnsafePointer[Float32, MutExternalOrigin],
    chosen_ref_logps: UnsafePointer[Float32, MutExternalOrigin],
    rejected_ref_logps: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    beta: Float32,
    B: Int32,
) raises:
    var block_size: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(B, block_size)
    var kernel = ctx.compile_function[dpo_loss_kernel, dpo_loss_kernel]()
    ctx.enqueue_function(
        kernel,
        chosen_logps,
        rejected_logps,
        chosen_ref_logps,
        rejected_ref_logps,
        output,
        beta,
        B,
        grid_dim=num_blocks,
        block_dim=block_size,
    )
    ctx.synchronize()
