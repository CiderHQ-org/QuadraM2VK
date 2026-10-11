// `air.calculate_unclamped_lod_texture_2d` (3 corpus sources) had no authored case, and it cannot
// have one in a kernel: it needs fragment implicit derivatives, the translator refuses it outside a
// fragment stage, and all three corpus sources that call it are fragment shaders.
//
// The coordinate is driven by `position`, not by a varying. A fragment case cannot choose its
// varyings -- the harness generates the vertex shader -- and the `uv` that arrives through the
// rasteriser is not bit-exactly `(x + 0.5) / W`. `gl_FragCoord` is: its derivative across a quad is
// exactly one pixel, so `c` advances by exactly 1/16 per pixel and each LOD below is an exact
// integer rather than something interpolated.
//
// Four textures at four sizes turn one derivative into four answers. The scale in texels per pixel
// is `size / 16`, so the LODs are log2 of 1, 4, 1/4 and 2 -- that is 0, 2, **-2** and 1. The
// negative one is the whole point of the symbol: `calculate_clamped_lod` would answer 0 there.
#include <metal_stdlib>
using namespace metal;

constexpr sampler mip_linear(coord::normalized, address::clamp_to_edge,
                             filter::linear, mip_filter::linear);

fragment float4 fragment_unclamped_lod_at_four_scales(
    float4 position [[position]],
    texture2d<float> t16 [[texture(0)]],
    texture2d<float> t64 [[texture(1)]],
    texture2d<float> t04 [[texture(2)]],
    texture2d<float> t32 [[texture(3)]])
{
    // The coordinate is driven by `position`, not by a varying: gl_FragCoord's derivative across a
    // quad is exactly 1 pixel, so `c` advances by exactly 1/16 per pixel and every LOD below is an
    // exact integer rather than something the rasteriser interpolated.
    float2 c = position.xy * (1.0 / 16.0);
    return float4(t16.calculate_unclamped_lod(mip_linear, c),
                  t64.calculate_unclamped_lod(mip_linear, c),
                  t04.calculate_unclamped_lod(mip_linear, c),
                  t32.calculate_unclamped_lod(mip_linear, c));
}
