// Each thread encodes one compute command -- a kernel buffer, a threadgroup length, a
// threadgroup dispatch, and a barrier -- and writes an output the command does not depend on.
// Both executors run the encoding under the authored descriptor policy; the encoded commands are
// not observed (no Metal readback; replay is owed), so a case observes the output buffer only.
#include <metal_stdlib>
using namespace metal;

struct Commands {
    command_buffer commands;
};

kernel void kernel_icb_encode_compute_commands(
    device const Commands &encoder [[buffer(0)]],
    device uint *data [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    compute_command command(encoder.commands, tid);
    command.set_kernel_buffer(data, 0);
    command.set_threadgroup_memory_length(16, 0);
    command.concurrent_dispatch_threadgroups(uint3(tid + 1, 1, 1), uint3(32, 1, 1));
    command.set_barrier();
    out[tid] = data[tid] * 3 + 7;
}
