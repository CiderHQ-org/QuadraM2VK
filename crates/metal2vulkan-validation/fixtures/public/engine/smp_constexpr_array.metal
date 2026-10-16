// Owned fixture: a fragment function that samples one element of a
// 32-texture array, picked per fragment, through one constexpr sampler, the shape of the corpus
// rows the translator handles. Where its 32 combined image samplers exceed the stage's samplers
// (moltenvk-m3 and lunarg-desktop-baseline-2023 have 16) the stage takes them apart into 32
// separate textures and one separate sampler (`sampler.separate_binding`); elsewhere they stay
// combined, and the vk12 profiles, whose 16 sampled images the 32 textures exceed in either
// form, refuse it. Pixel (x, y) of an 8 x 4 target samples texture x + 8y, of one texel, at its
// centre through a linear, clamp-to-edge sampler (`sampler.sample_hoisted_grad.switch`), so
// target(x, y) = that texel. Loop-free.
#include <metal_stdlib>
using namespace metal;

fragment half4 smp_constexpr_array(float4 pos [[position]],
                                   array<texture2d<half>, 32> atlas [[texture(0)]]) {
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    uint slice = (uint(pos.x) + 8 * uint(pos.y)) & 31;
    return atlas[slice].sample(s, float2(0.5));
}
