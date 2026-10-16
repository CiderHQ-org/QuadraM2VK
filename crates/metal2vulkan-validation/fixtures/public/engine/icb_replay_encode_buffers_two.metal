// Owned fixture: a kernel that encodes only the kernel buffers of the
// commands of two indirect command buffers whose other state the host encodes. Thread `tid` sets
// command `tid` of the first buffer to the constant `params` window and the device `out` window
// at 256 * tid bytes, and command `tid` of the second to the same `params` window and the `out`
// window at 256 * (tid + 2) bytes, as kernel buffers 0 and 1. Only executing the commands observes
// them. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer first;
    command_buffer second;
};

kernel void icb_replay_encode_buffers_two(
    device const Encoder &encoder [[buffer(0)]],
    constant uint *params [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    compute_command a(encoder.first, tid);
    a.set_kernel_buffer(params + tid * 64, 0);
    a.set_kernel_buffer(out + tid * 64, 1);
    compute_command b(encoder.second, tid);
    b.set_kernel_buffer(params + tid * 64, 0);
    b.set_kernel_buffer(out + (tid + 2) * 64, 1);
}
