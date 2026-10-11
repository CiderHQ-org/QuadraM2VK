#include <metal_stdlib>
using namespace metal;

kernel void kernel_dispatch_threads_boundary_barrier(
    device uint *output [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 local_size [[threads_per_threadgroup]])
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint index = gid.y * 10 + gid.x;
    output[index] = local_size.y * 100 + local_size.x;
}
