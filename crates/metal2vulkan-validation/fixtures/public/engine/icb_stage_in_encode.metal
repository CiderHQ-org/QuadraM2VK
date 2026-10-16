// Owned fixture: a kernel that encodes the stage-in region and the dispatch
// of commands whose pipeline and buffers the executing encoder supplies (inherited). Thread `tid`
// encodes command `tid`: the region of origin `origins[tid].xyz` and size `sizes[tid].xyz`, then
// a dispatch of `origins[tid].w` threads on x in threadgroups of `sizes[tid].w`. `out` and `data`
// are the buffers the executing encoder binds for the commands; this kernel only declares them.
// Only executing the commands observes them. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

kernel void icb_stage_in_encode(
    device const Encoder &encoder [[buffer(0)]],
    constant uint4 *origins [[buffer(1)]],
    constant uint4 *sizes [[buffer(2)]],
    device uint4 *out [[buffer(3)]],
    device const uint *data [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    compute_command command(encoder.commands, tid);
    uint4 origin = origins[tid];
    uint4 size = sizes[tid];
    command.set_stage_in_region(origin.xyz, size.xyz);
    command.concurrent_dispatch_threads(uint3(origin.w, 1, 1), uint3(size.w, 1, 1));
}
