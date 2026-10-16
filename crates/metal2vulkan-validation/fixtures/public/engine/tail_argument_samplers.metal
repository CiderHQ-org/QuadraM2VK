// Owned MSL for argument-buffer samplers: a sampler loaded from an
// argument buffer beside the texture it samples, in forms whose answer is exact on Metal, so
// Metal's output is the comparison's reference.
#include <metal_stdlib>
using namespace metal;

struct Material {
    texture2d<float> t;
    sampler s;
};

// Thread g samples the material's texture at (0.25g + 0.125, 0.25g + 0.125) in the sampler's
// coordinates, level 0, through the material's sampler, and writes the texel. The sampler's
// coordinate mode, filter, and address mode are the case's.
kernel void argument_sampler_sample(constant Material &m [[buffer(0)]],
                                    device float4 *out [[buffer(1)]],
                                    uint g [[thread_position_in_grid]])
{
    float c = 0.25 * float(g) + 0.125;
    out[g] = m.t.sample(m.s, float2(c, c), level(0.0));
}

// The same through a pixel coordinate on texel edges: thread g samples at (g / 4, g / 4), so a
// pixel sampler reads on an edge for every g that is a multiple of 4.
kernel void argument_sampler_edges(constant Material &m [[buffer(0)]],
                                   device float4 *out [[buffer(1)]],
                                   uint g [[thread_position_in_grid]])
{
    float c = 0.25 * float(g);
    out[g] = m.t.sample(m.s, float2(c, c), level(0.0));
}

struct DepthMaterial {
    depth2d<float> d;
    sampler s;
};

// Thread g compares the material's depth texture at (0.25g, 0.25g), level 0, against 0.5
// through the material's comparison sampler.
kernel void argument_sampler_compare(constant DepthMaterial &m [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     uint g [[thread_position_in_grid]])
{
    float c = 0.25 * float(g);
    out[g] = m.d.sample_compare(m.s, float2(c, c), 0.5, level(0.0));
}

struct CountMaterial {
    texture2d<uint> t;
    sampler s;
};

// Thread g samples the material's unsigned texture at (0.25g + 0.125, 0.25g + 0.125), level 0.
kernel void argument_sampler_integer(constant CountMaterial &m [[buffer(0)]],
                                     device uint4 *out [[buffer(1)]],
                                     uint g [[thread_position_in_grid]])
{
    float c = 0.25 * float(g) + 0.125;
    out[g] = m.t.sample(m.s, float2(c, c), level(0.0));
}

struct VaryingIn {
    float4 position [[position]];
    float2 uv;
};

// A fragment's implicit-LOD sample of the material's texture through its sampler.
fragment float4 argument_sampler_fragment(VaryingIn in [[stage_in]],
                                          constant Material &m [[buffer(0)]])
{
    return m.t.sample(m.s, in.uv);
}
