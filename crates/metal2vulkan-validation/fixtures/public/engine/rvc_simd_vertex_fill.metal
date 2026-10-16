// Owned fixture (written for this repository): simd_shuffle_and_fill_down
// with an explicit modulo of 32 in a vertex function. Metal runs this form there; its compiler
// refuses the form without a modulo in a vertex function (it reads the simdgroup width,
// air.get_simdgroup_size, which a vertex function may not). Forty rasterization-disabled points;
// vertex v writes simd_shuffle_and_fill_down(in[v], -in[v], 3, 32) to out[v]. Loop-free.
#include <metal_stdlib>
using namespace metal;

vertex void simd_shuffle_and_fill_down_in_a_vertex_function(uint vid [[vertex_id]],
                                                            const device float *in [[buffer(0)]],
                                                            device uint *out [[buffer(1)]])
{
    const float x = in[vid];
    out[vid] = as_type<uint>(simd_shuffle_and_fill_down(x, -x, ushort(3), ushort(32)));
}
