// Owned fixture (written for this repository): simdgroup functions in a
// vertex function. Forty rasterization-disabled points; vertex v reads in[v] and writes simd_sum,
// simd_shuffle from lane 5, simd_shuffle_xor by 1, and the ballot of x > 0 (low and high words) to
// out[5v .. 5v + 4]. Metal runs them in a vertex function; the translation needs a device that
// supports subgroup operations in the vertex stage. Loop-free.
#include <metal_stdlib>
using namespace metal;

vertex void simd_functions_in_a_vertex_function(uint vid [[vertex_id]],
                                                const device float *in [[buffer(0)]],
                                                device uint *out [[buffer(1)]])
{
    const float x = in[vid];
    const float sum = simd_sum(x);
    const float from5 = simd_shuffle(x, ushort(5));
    const float xored = simd_shuffle_xor(x, ushort(1));
    const ulong bits = ulong(simd_vote::vote_t(simd_ballot(x > 0.0f)));
    out[vid * 5 + 0] = as_type<uint>(sum);
    out[vid * 5 + 1] = as_type<uint>(from5);
    out[vid * 5 + 2] = as_type<uint>(xored);
    out[vid * 5 + 3] = uint(bits);
    out[vid * 5 + 4] = uint(bits >> 32);
}
