// Owned fixture: constexpr samplers with anisotropy and no mip filter, sampled with gradients.
// Vulkan has no sampler without mipmapping, so such a sampler is a nearest mip filter with a level
// clamp (`metal::sampler::vk_sampler`); through Metal's own sampler (MoltenVK) an anisotropic one
// whose minification and magnification filters differ reads no clamp as Metal's `NotMipmapped`.
// - `anisotropic_mixed_filters`: linear minification, nearest magnification, `max_anisotropy(8)`,
//   no mip filter: recorded as `NotMipmappedAnisotropicFilter` through Metal's own sampler.
// - `anisotropic_equal_filters`: both filters linear: no note.
// - `anisotropic_mipmapped_mixed`: the first one's filters with a linear mip filter: no note.
// Lane `i` reads its coordinates and gradient lengths from `in[i]`. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void anisotropic_mixed_filters(texture2d<float> t [[texture(0)]],
                                      device const float4 *in [[buffer(0)]],
                                      device float4 *out [[buffer(1)]],
                                      uint i [[thread_position_in_grid]])
{
    constexpr sampler s(coord::normalized, min_filter::linear, mag_filter::nearest,
                        mip_filter::none, address::clamp_to_edge, max_anisotropy(8));
    const float4 g = in[i];
    out[i] = t.sample(s, g.xy, gradient2d(float2(g.z, 0.0f), float2(0.0f, g.w)));
}

kernel void anisotropic_equal_filters(texture2d<float> t [[texture(0)]],
                                      device const float4 *in [[buffer(0)]],
                                      device float4 *out [[buffer(1)]],
                                      uint i [[thread_position_in_grid]])
{
    constexpr sampler s(coord::normalized, filter::linear, mip_filter::none,
                        address::clamp_to_edge, max_anisotropy(8));
    const float4 g = in[i];
    out[i] = t.sample(s, g.xy, gradient2d(float2(g.z, 0.0f), float2(0.0f, g.w)));
}

kernel void anisotropic_mipmapped_mixed(texture2d<float> t [[texture(0)]],
                                        device const float4 *in [[buffer(0)]],
                                        device float4 *out [[buffer(1)]],
                                        uint i [[thread_position_in_grid]])
{
    constexpr sampler s(coord::normalized, min_filter::linear, mag_filter::nearest,
                        mip_filter::linear, address::clamp_to_edge, max_anisotropy(8));
    const float4 g = in[i];
    out[i] = t.sample(s, g.xy, gradient2d(float2(g.z, 0.0f), float2(0.0f, g.w)));
}
