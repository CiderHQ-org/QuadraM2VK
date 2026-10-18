// Owned fixture: a fragment function's `calculate_clamped_lod` through constexpr samplers
// whose min and mag filters differ, beside one whose filters agree. Metal clamps the level at its
// grid's bounds whatever the filters: a constexpr clamp is read as a half, then a linear mip
// filter's bound is truncated onto sixty-fourths and a nearest one's is the level it reads there.
//
// The coordinate is driven by `position`, as in `fragment_unclamped_lod_at_four_scales`: its
// derivative across a quad is exactly `k` per pixel, so on the 16x16 texture each band of four
// rows has the level log2(16 k) exactly: -2, 2, 4 and 6 from the top. The texture has five levels.
#include <metal_stdlib>
using namespace metal;

// Min linear, mag nearest; clamp [half(1.3), half(2.7)], on the grid [83/64, 172/64].
constexpr sampler tib_min_linear(coord::normalized, address::repeat, min_filter::linear,
                                 mag_filter::nearest, mip_filter::linear, lod_clamp(1.3, 2.7));
// Min nearest, mag linear, nearest mip filter; clamp [half(2.505), 16], on the grid level 2 to 16.
constexpr sampler tib_min_nearest(coord::normalized, address::repeat, min_filter::nearest,
                                  mag_filter::linear, mip_filter::nearest, lod_clamp(2.505, 16.0));
// Both filters linear: the control.
constexpr sampler tib_both_linear(coord::normalized, address::repeat, filter::linear,
                                  mip_filter::linear, lod_clamp(1.3, 2.7));

fragment float4 tib_clamped_lod_mixed_filters(float4 position [[position]],
                                              texture2d<float> t [[texture(0)]])
{
    uint band = uint(position.y) / 4;
    float k = band == 0 ? 1.0 / 64.0 : band == 1 ? 0.25 : band == 2 ? 1.0 : 4.0;
    float2 c = position.xy * k;
    return float4(t.calculate_clamped_lod(tib_min_linear, c),
                  t.calculate_clamped_lod(tib_min_nearest, c),
                  t.calculate_clamped_lod(tib_both_linear, c),
                  t.calculate_unclamped_lod(tib_min_linear, c));
}
