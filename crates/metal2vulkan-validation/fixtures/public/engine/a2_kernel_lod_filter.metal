// Owned fixture: level of detail in a kernel, read through the choice
// between the magnification and minification filters. The sampler magnifies
// with `nearest` and minifies with `linear`, so a sample 1/4 texel past a texel centre returns the
// texel alone when its level of detail is at most 0 and a bilinear blend when it is above 0. The
// texture has one level, so this reads the sign of the level of detail, not the level itself.
// Each lane samples four ways: implicit (no level, bias, clamp, or gradient), with `bias`, with
// `min_lod_clamp`, and with both; the coordinate, bias, and clamp come from the input buffer.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler mag_nearest_min_linear(mag_filter::nearest, min_filter::linear,
                                        address::clamp_to_edge, coord::normalized);

kernel void a2_kernel_lod_filter(texture2d<float, access::sample> t [[texture(0)]],
                                 const device float4 *params [[buffer(0)]],
                                 device float4 *out [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]])
{
    float4 q = params[tid];
    float2 uv = q.xy;
    out[tid] = float4(t.sample(mag_nearest_min_linear, uv).x,
                      t.sample(mag_nearest_min_linear, uv, bias(q.z)).x,
                      t.sample(mag_nearest_min_linear, uv, min_lod_clamp(q.w)).x,
                      t.sample(mag_nearest_min_linear, uv, bias(q.z), min_lod_clamp(q.w)).x);
}
