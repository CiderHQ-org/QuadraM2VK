// Owned fixture: a kernel that encodes only the kernel buffers of
// commands whose other state the host encodes. Thread `tid` sets command `tid`'s kernel buffers
// 0 and 1 to the constant `params` window and the device `out` window at 256 * tid bytes; the
// pipeline is inherited from the executing encoder, and each dispatch and threadgroup length is
// the host's. Only executing the commands observes them. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

kernel void icb_replay_encode_buffers(
    device const Encoder &encoder [[buffer(0)]],
    constant uint *params [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    compute_command command(encoder.commands, tid);
    command.set_kernel_buffer(params + tid * 64, 0);
    command.set_kernel_buffer(out + tid * 64, 1);
}
