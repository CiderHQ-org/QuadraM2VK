// Owned fixture: a kernel whose only effect is the compute commands it
// encodes. Thread `tid` encodes command `tid`: the pipeline state of its argument buffer, the
// constant `params` window and the device `out` window at 256 * tid bytes as kernel buffers 0
// and 1, a 32-byte threadgroup length, a dispatch of 64 threads in groups of 8 (as threadgroups
// for an even command, as threads for an odd one), and a barrier. Only executing the commands
// observes them. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
    compute_pipeline_state pipeline;
};

kernel void icb_replay_encode(
    device const Encoder &encoder [[buffer(0)]],
    constant uint *params [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    compute_command command(encoder.commands, tid);
    command.set_compute_pipeline_state(encoder.pipeline);
    command.set_kernel_buffer(params + tid * 64, 0);
    command.set_kernel_buffer(out + tid * 64, 1);
    command.set_threadgroup_memory_length(32, 0);
    if ((tid & 1) == 0) {
        command.concurrent_dispatch_threadgroups(uint3(8, 1, 1), uint3(8, 1, 1));
    } else {
        command.concurrent_dispatch_threads(uint3(64, 1, 1), uint3(8, 1, 1));
    }
    command.set_barrier();
}
