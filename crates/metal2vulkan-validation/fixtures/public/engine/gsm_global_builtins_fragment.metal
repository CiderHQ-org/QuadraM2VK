// Owned fixture (written for this repository): Metal 3.1's
// program-scope built-in variables in a fragment function, which may declare
// thread_index_in_quadgroup, threads_per_simdgroup and thread_index_in_simdgroup (Apple's
// compiler refuses the other kernel built-ins there, and every one in a vertex function). The
// fragment writes its simdgroup's width, read from a global uint threads_per_simdgroup, to an
// r32_uint target. Loop-free.
// metal-std: metal3.1
#include <metal_stdlib>
using namespace metal;

uint simd_width [[threads_per_simdgroup]];

fragment uint global_builtins_in_a_fragment(float4 position [[position]])
{
    return simd_width;
}
