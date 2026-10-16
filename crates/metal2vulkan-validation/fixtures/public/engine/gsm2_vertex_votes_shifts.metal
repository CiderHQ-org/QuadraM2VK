// Owned fixture (written for this repository): simdgroup votes and relative
// shuffles in a vertex function. Rasterization-disabled points; vertex v reads in[v].
// simd_votes_in_a_vertex_function writes simd_all and simd_any of x > 0 and of x > 100 to
// out[4v .. 4v + 3]; simd_shifts_in_a_vertex_function writes simd_shuffle_up by 3,
// simd_shuffle_down by 5 and simd_shuffle_and_fill_up(x, -x, 2, 32) to out[3v .. 3v + 2]. Metal
// runs them in a vertex function (the fill with an explicit modulo, the form its compiler takes
// there); the translation needs a device that supports subgroup operations in the vertex stage.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

vertex void simd_votes_in_a_vertex_function(uint vid [[vertex_id]],
                                            const device float *in [[buffer(0)]],
                                            device uint *out [[buffer(1)]])
{
    const float x = in[vid];
    out[vid * 4 + 0] = uint(simd_all(x > 0.0f));
    out[vid * 4 + 1] = uint(simd_any(x > 0.0f));
    out[vid * 4 + 2] = uint(simd_all(x > 100.0f));
    out[vid * 4 + 3] = uint(simd_any(x > 100.0f));
}

vertex void simd_shifts_in_a_vertex_function(uint vid [[vertex_id]],
                                             const device float *in [[buffer(0)]],
                                             device uint *out [[buffer(1)]])
{
    const float x = in[vid];
    out[vid * 3 + 0] = as_type<uint>(simd_shuffle_up(x, ushort(3)));
    out[vid * 3 + 1] = as_type<uint>(simd_shuffle_down(x, ushort(5)));
    out[vid * 3 + 2] = as_type<uint>(simd_shuffle_and_fill_up(x, -x, ushort(2), ushort(32)));
}
