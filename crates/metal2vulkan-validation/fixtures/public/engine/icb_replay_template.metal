// Owned fixture: the compute template an executed indirect command
// dispatches. Each thread stages `params[0] * (tid + 1)` in threadgroup memory, waits for its
// group, and writes its own staged word plus `params[1]` to `out[tid]`. A command binds `params`
// and `out` (each at an offset into a larger buffer) and sets the threadgroup length. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void icb_replay_template(
    constant uint *params [[buffer(0)]],
    device uint *out [[buffer(1)]],
    threadgroup uint *scratch [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    scratch[lid] = params[0] * (tid + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[tid] = scratch[lid] + params[1];
}
