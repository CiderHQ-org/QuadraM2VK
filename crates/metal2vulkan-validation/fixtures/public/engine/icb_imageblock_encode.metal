// Owned fixture: a kernel that encodes the imageblock size, kernel buffer
// and dispatch of commands whose pipeline the executing encoder supplies (inherited). Thread `tid`
// encodes command `tid`: kernel buffer 0 the `out` window at 1024 * tid bytes, the imageblock
// size `shapes[tid].xy`, and a dispatch of `grids[tid].xy` threads in threadgroups of
// `shapes[tid].zw`. Only executing the commands observes them. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

kernel void icb_imageblock_encode(
    device const Encoder &encoder [[buffer(0)]],
    constant ushort4 *shapes [[buffer(1)]],
    constant uint2 *grids [[buffer(2)]],
    device uint4 *out [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    compute_command command(encoder.commands, tid);
    ushort4 shape = shapes[tid];
    uint2 grid = grids[tid];
    command.set_kernel_buffer(out + 64 * tid, 0);
    command.set_imageblock_size(shape.xy);
    command.concurrent_dispatch_threads(uint3(grid, 1), uint3(uint(shape.z), uint(shape.w), 1));
}
