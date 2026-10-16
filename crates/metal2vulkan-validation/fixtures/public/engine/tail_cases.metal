// Owned MSL for authored cases: the classes the translator implements, each in a
// form whose answer is exact on every device, so Metal's output is the comparison's reference.
#include <metal_stdlib>
using namespace metal;

constexpr sampler mip_linear(coord::normalized, address::clamp_to_edge, filter::linear,
                             mip_filter::linear);

// An LOD query through a texture selected per fragment: each candidate's query runs
// before the switch-at-use, which only selects the value. The coordinate is driven by
// `position`, whose derivative across a quad is exactly one pixel, so `c` advances by exactly
// 1/16 per pixel and each candidate's unclamped LOD is an exact integer (log2 of size / 16). The
// selection splits the target at x = 8, between quads, so every quad reads one candidate.
fragment float4 tail_select_unclamped_lod(float4 position [[position]],
                                          texture2d<float> near [[texture(0)]],
                                          texture2d<float> far [[texture(1)]])
{
    texture2d<float> t = position.x < 8.0 ? near : far;
    float2 c = position.xy * (1.0 / 16.0);
    return float4(t.calculate_unclamped_lod(mip_linear, c),
                  t.calculate_clamped_lod(mip_linear, c), 0.0, 1.0);
}

constexpr sampler cube_nearest(coord::normalized, address::clamp_to_edge, filter::nearest);

struct Cubes {
    texturecube<float> sky;
    texturecube_array<float> skies;
};

// An argument buffer's cube sampled and queried through its cube view, and a cube array only
// queried. Thread g samples the face along axis g / 2, positive for an even g, at its
// center; each face holds one value, so the sample pins the face alone.
kernel void tail_heap_cube_queries(constant Cubes &cubes [[buffer(0)]],
                                   device float4 *out [[buffer(1)]],
                                   uint g [[thread_position_in_grid]])
{
    float sign = (g & 1) != 0 ? -1.0 : 1.0;
    float3 d = float3(g / 2 == 0 ? sign : 0.0, g / 2 == 1 ? sign : 0.0, g / 2 == 2 ? sign : 0.0);
    out[g] = float4(cubes.sky.sample(cube_nearest, d, level(0.0)).x,
                    float(cubes.sky.get_num_mip_levels()), float(cubes.sky.get_width()),
                    float(cubes.skies.get_array_size()));
}
