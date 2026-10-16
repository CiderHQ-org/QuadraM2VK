// Owned MSL for a cube gradient through an argument buffer's sampler: a
// cube sampled with explicit gradients through a sampler loaded from an argument buffer beside
// it, in a form whose answer is exact on Metal, so Metal's output is the comparison's reference.
#include <metal_stdlib>
using namespace metal;

struct CubeMaterial {
    texturecube<float> t;
    sampler s;
};

// Thread g samples the material's cube on its +x face, at the direction (1, h, h) with h = 0.25
// for g < 2 and -0.25 otherwise, through the material's sampler with explicit cube gradients:
// zero for an even g, and dP/dx = (0, 0, 4), dP/dy = (0, 4, 0) for an odd g. A zero gradient
// magnifies; the odd gradient spans four texels of a 2x2 face, so it minifies far from the
// boundary. A sampler that minifies linearly and magnifies nearest tells the two apart.
kernel void argument_sampler_cube_gradient(constant CubeMaterial &m [[buffer(0)]],
                                           device float4 *out [[buffer(1)]],
                                           uint g [[thread_position_in_grid]])
{
    float h = g < 2 ? 0.25 : -0.25;
    float k = float(g & 1) * 4.0;
    out[g] = m.t.sample(m.s, float3(1.0, h, h), gradientcube(float3(0.0, 0.0, k), float3(0.0, k, 0.0)));
}
