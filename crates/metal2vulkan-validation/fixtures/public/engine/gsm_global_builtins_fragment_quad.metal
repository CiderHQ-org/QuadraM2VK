// Owned fixture (written for this repository): Metal 3.1's
// program-scope thread_index_in_quadgroup in a fragment function, as a global uint. The fragment
// writes its quad lane to an r32_uint target. Apple's compiler accepts it; whether the Metal device
// lowers it is what the M4 qualification measures (a global ushort one was refused at pipeline
// creation). Loop-free.
// metal-std: metal3.1
#include <metal_stdlib>
using namespace metal;

uint quad_lane [[thread_index_in_quadgroup]];

fragment uint global_quad_lane_in_a_fragment(float4 position [[position]])
{
    return quad_lane;
}
