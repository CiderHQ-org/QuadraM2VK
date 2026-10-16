// Owned fixture: a fragment function that samples two 16-texture arrays
// through one constexpr sampler. Where their 32 combined image samplers exceed the stage's
// samplers (moltenvk-m3 and lunarg-desktop-baseline-2023 have 16) both arrays share the stage's
// one separate sampler; elsewhere they stay combined, and the vk12 profiles, whose 16 sampled
// images the 32 textures exceed in either form, refuse it. Pixel (x, y) of a 16 x 16
// target adds texel 0 of a[x] and of b[y], each of one texel sampled at its centre through a
// nearest, clamp-to-edge sampler (two `sampler.sample_hoisted_grad.switch`es). Loop-free.
#include <metal_stdlib>
using namespace metal;

fragment float4 smp_shared_sampler(float4 pos [[position]],
                                   array<texture2d<float>, 16> a [[texture(0)]],
                                   array<texture2d<float>, 16> b [[texture(16)]]) {
    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);
    uint x = uint(pos.x) & 15;
    uint y = uint(pos.y) & 15;
    return a[x].sample(s, float2(0.5)) + b[y].sample(s, float2(0.5));
}
