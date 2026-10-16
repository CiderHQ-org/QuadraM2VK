// Eight physical coordinates mapped to screen coordinates through a rasterization rate map whose
// rate is a uniform 1.0.
//
// `air.map_physical_to_screen_coordinates.v2f32.p2i8.i32` (8 case-less corpus sources) -- the
// direction opposite the one `blit-through-a-uniform-rasterization-rate-map` pinned.
//
// The old engine lowers the call to the identity, which is correct for a map whose every
// horizontal and vertical rate is 1.0 and whose physical size equals its screen size; its
// reflection reports buffer 0 Unused. The engine decodes buffer 0 as this crate's parameter
// data, so the case authors buffer 0 as the map itself (`rasterization_rate_map`:
// 64x64 screen, one layer, eight rates of 1.0 on each axis): the Metal oracle binds what
// `MTLRasterizationRateMap.copyParameterDataToBuffer:offset:` writes for it, and a Vulkan executor
// binds `bind::RasterizationRateMap`'s data of the same map. Both implementations compile it to
// the identity.
//
// The eight coordinates are chosen so that a wrong answer cannot hide.
//
//   * Four are INSIDE the map -- (0, 0), (1.5, 2.5), (31.25, 47.75), (63.5, 63.5) -- spanning the
//     origin, the far corner and two interior points whose fractions are quarters. A decoder that
//     rounded to texel centres or floored to integers moves three of them.
//   * Three are OUTSIDE it, which is what the corpus never asks: (64, 64) is exactly one past the
//     far edge, (80, 7.25) is sixteen past it on one axis only, and (-3.5, 2) is negative on one
//     axis only. These separate the identity from a clamp to the map's extent, which would answer
//     63-and-a-bit, 63-and-a-bit and 0 where the identity answers 64, 80 and -3.5.
//   * The last, (8.125, 55.875), is asymmetric in both axes, so a transposed mapping is visible.
//
// Every coordinate is an exact binary fraction below 128, so all sixteen output words are exact in
// binary32 and the comparison needs no tolerance.
//
// **The identity is not Metal's answer everywhere, and the boundary is measured.** On this map
// Metal is exactly the identity out to x = 95, and at x = 96 it returns 0 -- not a clamp to 63.x,
// which is why "outside the extent" alone does not describe it. 100.5 was in an earlier draft of
// this file and Metal answered 0.0 for it; the bisect that followed put the break between 95 and
// 96, one and a half times the 64-wide screen. So the case stays inside the domain the identity is
// measured to hold on, and the divergence past it is recorded rather than papered over: a shader
// that asks this call for a coordinate more than half a screen past the physical extent will get a
// different answer from us than from Metal, and none of the eight corpus sources does.
#include <metal_stdlib>
using namespace metal;

constant float2 kPhysical[8] = {
    float2(0.0f, 0.0f),
    float2(1.5f, 2.5f),
    float2(31.25f, 47.75f),
    float2(63.5f, 63.5f),
    float2(64.0f, 64.0f),
    float2(80.0f, 7.25f),
    float2(-3.5f, 2.0f),
    float2(8.125f, 55.875f),
};

kernel void kernel_rate_map_physical_to_screen(
    constant rasterization_rate_map_data &map [[buffer(0)]],
    device uint *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    rasterization_rate_map_decoder decoder(map);
    float2 screen = decoder.map_physical_to_screen_coordinates(kPhysical[tid]);
    out[tid * 2 + 0] = as_type<uint>(screen.x);
    out[tid * 2 + 1] = as_type<uint>(screen.y);
}
