// Owned fixture (written for this repository): Metal 3.1's
// program-scope built-in variables in a kernel. Every kernel built-in is declared as a global
// variable (each vector one as uint3) and read in the kernel, the simdgroup and quad lanes in a
// helper Metal keeps out of line. Thread t of the grid (t = x + width * (y + height * z) of its
// thread_position_in_grid over threads_per_grid) writes 29 words from out[29 t]:
// threads_per_threadgroup (3 words), thread_position_in_threadgroup (3),
// thread_index_in_threadgroup, thread_index_in_simdgroup, simdgroup_index_in_threadgroup,
// simdgroups_per_threadgroup, thread_index_in_quadgroup, quadgroup_index_in_threadgroup,
// quadgroups_per_threadgroup, thread_position_in_grid (3), threadgroup_position_in_grid (3),
// threads_per_grid (3), threadgroups_per_grid (3), dispatch_threads_per_threadgroup (3) and
// dispatch_simdgroups_per_threadgroup. Loop-free.
// metal-std: metal3.1
#include <metal_stdlib>
using namespace metal;

uint3 group_size [[threads_per_threadgroup]];
uint3 position_in_group [[thread_position_in_threadgroup]];
uint index_in_group [[thread_index_in_threadgroup]];
uint simd_lane [[thread_index_in_simdgroup]];
uint simd_index [[simdgroup_index_in_threadgroup]];
uint simd_count [[simdgroups_per_threadgroup]];
uint quad_lane [[thread_index_in_quadgroup]];
uint quad_index [[quadgroup_index_in_threadgroup]];
uint quad_count [[quadgroups_per_threadgroup]];
uint3 position_in_grid [[thread_position_in_grid]];
uint3 group_position [[threadgroup_position_in_grid]];
uint3 grid_size [[threads_per_grid]];
uint3 group_count [[threadgroups_per_grid]];
uint3 nominal_group_size [[dispatch_threads_per_threadgroup]];
uint nominal_simd_count [[dispatch_simdgroups_per_threadgroup]];

__attribute__((noinline)) static void write_lanes(device uint *o)
{
    o[7] = simd_lane;
    o[10] = quad_lane;
}

kernel void global_builtins_in_a_kernel(device uint *out [[buffer(0)]])
{
    const uint t =
        position_in_grid.x + grid_size.x * (position_in_grid.y + grid_size.y * position_in_grid.z);
    device uint *o = out + 29 * t;
    o[0] = group_size.x;
    o[1] = group_size.y;
    o[2] = group_size.z;
    o[3] = position_in_group.x;
    o[4] = position_in_group.y;
    o[5] = position_in_group.z;
    o[6] = index_in_group;
    write_lanes(o);
    o[8] = simd_index;
    o[9] = simd_count;
    o[11] = quad_index;
    o[12] = quad_count;
    o[13] = position_in_grid.x;
    o[14] = position_in_grid.y;
    o[15] = position_in_grid.z;
    o[16] = group_position.x;
    o[17] = group_position.y;
    o[18] = group_position.z;
    o[19] = grid_size.x;
    o[20] = grid_size.y;
    o[21] = grid_size.z;
    o[22] = group_count.x;
    o[23] = group_count.y;
    o[24] = group_count.z;
    o[25] = nominal_group_size.x;
    o[26] = nominal_group_size.y;
    o[27] = nominal_group_size.z;
    o[28] = nominal_simd_count;
}
