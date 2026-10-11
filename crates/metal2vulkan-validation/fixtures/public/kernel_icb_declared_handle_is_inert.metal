// An argument buffer that declares an indirect command buffer the kernel never encodes. The
// command buffer is an inert resource both executors supply; the kernel's observable effect is
// its output buffer, computed from the argument buffer's other member.
#include <metal_stdlib>
using namespace metal;

struct Commands {
    command_buffer commands;
    uint scale;
};

kernel void kernel_icb_declared_handle_is_inert(
    device const Commands &encoder [[buffer(0)]],
    device uint *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = tid * encoder.scale + 1;
}
