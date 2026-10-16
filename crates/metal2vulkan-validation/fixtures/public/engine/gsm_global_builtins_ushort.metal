// Owned fixture (written for this repository): Metal 3.1's
// program-scope built-in variables of ushort types in a kernel, which Apple's compiler reads with
// the `.i16` calls (`air.get_local_size.i16(i16 1)`). Thread t of the grid (t = x + width * y of
// its ushort2 thread_position_in_grid over the uint3 threads_per_grid) writes 8 words from
// out[8 t]: its ushort3 threads_per_threadgroup (3 words), its thread_position_in_grid (2), and
// its ushort thread_index_in_threadgroup, thread_index_in_simdgroup and
// quadgroup_index_in_threadgroup. Loop-free.
// metal-std: metal3.1
#include <metal_stdlib>
using namespace metal;

ushort3 group_size [[threads_per_threadgroup]];
ushort2 position_in_grid [[thread_position_in_grid]];
ushort index_in_group [[thread_index_in_threadgroup]];
ushort simd_lane [[thread_index_in_simdgroup]];
ushort quad_index [[quadgroup_index_in_threadgroup]];
uint3 grid_size [[threads_per_grid]];

kernel void global_ushort_builtins_in_a_kernel(device uint *out [[buffer(0)]])
{
    const uint t = uint(position_in_grid.x) + grid_size.x * uint(position_in_grid.y);
    device uint *o = out + 8 * t;
    o[0] = group_size.x;
    o[1] = group_size.y;
    o[2] = group_size.z;
    o[3] = position_in_grid.x;
    o[4] = position_in_grid.y;
    o[5] = index_in_group;
    o[6] = simd_lane;
    o[7] = quad_index;
}
