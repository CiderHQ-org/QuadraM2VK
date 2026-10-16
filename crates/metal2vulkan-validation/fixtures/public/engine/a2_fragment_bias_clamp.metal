// Owned fixture: whether Metal clamps a large sample bias
// (Vulkan clamps the Bias operand to maxSamplerLodBias, 2 to 16 on the committed profiles;
// Metal documents no clamp). A fragment's implicit level of detail is log2 of its texel-space
// derivative, so a coordinate that advances 2^k normalized units per pixel over a 16-texel
// texture has level k + 4 before the bias. Each 2x2 quad reads its own (k, bias) pair; the
// sampler magnifies with `nearest` and minifies with `linear`, so the colour shows the sign of
// the biased level. Repeat addressing makes every pixel of a quad with k >= 0 sample the same
// texel phase. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler mag_nearest_min_linear_repeat(mag_filter::nearest, min_filter::linear,
                                               address::repeat, coord::normalized);

fragment float4 a2_fragment_bias_clamp(float4 pos [[position]],
                                       texture2d<float, access::sample> t [[texture(0)]],
                                       constant float2 *params [[buffer(0)]])
{
    uint2 pixel = uint2(pos.xy);
    float2 q = params[(pixel.y >> 1) * 4 + (pixel.x >> 1)];
    float2 uv = float2(5.75f, 5.75f) / 16.0f + float2(pixel) * q.x;
    return t.sample(mag_nearest_min_linear_repeat, uv, bias(q.y));
}
