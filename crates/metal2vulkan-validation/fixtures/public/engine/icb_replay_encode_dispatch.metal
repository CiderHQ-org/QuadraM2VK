// Owned fixture: a kernel that encodes only the dispatches of commands
// whose pipeline and buffers the executing encoder supplies (inherited). Thread `tid` encodes
// command `tid`: a 128-byte threadgroup length and a dispatch in threadgroups of 8 x 4 of the
// grid `grids[tid].xyz`, of threads when `grids[tid].w` is 0 and of threadgroups otherwise, with
// a barrier for every command but the first. `params` and `out` are the buffers the executing
// encoder binds for the commands; this kernel only declares them. Only executing the commands
// observes them. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

kernel void icb_replay_encode_dispatch(
    device const Encoder &encoder [[buffer(0)]],
    constant uint4 *grids [[buffer(1)]],
    constant uint *params [[buffer(2)]],
    device uint *out [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    compute_command command(encoder.commands, tid);
    command.set_threadgroup_memory_length(128, 0);
    uint4 g = grids[tid];
    if (g.w == 0) {
        command.concurrent_dispatch_threads(g.xyz, uint3(8, 4, 1));
    } else {
        command.concurrent_dispatch_threadgroups(g.xyz, uint3(8, 4, 1));
    }
    if (tid != 0) {
        command.set_barrier();
    }
}
