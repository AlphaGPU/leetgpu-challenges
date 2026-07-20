from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from std.math import ceildiv


def ppo_loss_kernel(
    advantages: UnsafePointer[Float32, MutExternalOrigin],
    log_pi: UnsafePointer[Float32, MutExternalOrigin],
    log_pi_old: UnsafePointer[Float32, MutExternalOrigin],
    log_ref: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    clip_eps: Float32,
    beta: Float32,
    N: Int32,
):
    pass


# advantages, log_pi, log_pi_old, log_ref, output are device pointers (i.e. pointers to memory on the GPU)
@export
def solve(
    advantages: UnsafePointer[Float32, MutExternalOrigin],
    log_pi: UnsafePointer[Float32, MutExternalOrigin],
    log_pi_old: UnsafePointer[Float32, MutExternalOrigin],
    log_ref: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    clip_eps: Float32,
    beta: Float32,
    B: Int32,
    S: Int32,
) raises:
    var block_size: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(B * S, block_size)
    var kernel = ctx.compile_function[ppo_loss_kernel, ppo_loss_kernel]()
    ctx.enqueue_function(
        kernel,
        advantages,
        log_pi,
        log_pi_old,
        log_ref,
        output,
        clip_eps,
        beta,
        B * S,
        grid_dim=num_blocks,
        block_dim=block_size,
    )
    ctx.synchronize()
