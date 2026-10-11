#include <metal_stdlib>
using namespace metal;

struct FragmentVaryingInput {
    float4 position [[position]];
    float2 uv;
};

fragment float4 fragment_varying_color(FragmentVaryingInput input [[stage_in]]) {
    return float4(input.uv, 0.5, 1.0);
}
