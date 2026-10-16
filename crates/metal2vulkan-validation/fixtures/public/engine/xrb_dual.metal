// Owned fixture: a blend of the second source into an extended-range
// attachment without fast math, for the second-source lane of the extended-range blend table
// (`metal2vulkan_devtools::xr_blends::dual_table`). The fullscreen triangle of `rt6_samples`;
// each fragment returns its output and a second one (`[[color(0), index(1)]]`), each read from its
// own buffer at the fragment's texel, `values[y * width + x]`, as `half4` or `float4`.
// metal-fast-math: off
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct XrbDualVertex {
    float4 position [[position]];
};

vertex XrbDualVertex xrb_dual_vertex(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return XrbDualVertex { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

struct XrbDualHalf {
    half4 color [[color(0), index(0)]];
    half4 second [[color(0), index(1)]];
};

struct XrbDualFloat {
    float4 color [[color(0), index(0)]];
    float4 second [[color(0), index(1)]];
};

// The fragment's texel in a row of `width` texels.
static uint texel(XrbDualVertex input, uint width) {
    return uint(input.position.y) * width + uint(input.position.x);
}

fragment XrbDualHalf xrb_dual_half(XrbDualVertex input [[stage_in]],
        const device half4 *colors [[buffer(0)]], constant uint &width [[buffer(1)]],
        const device half4 *seconds [[buffer(2)]]) {
    uint i = texel(input, width);
    return XrbDualHalf { colors[i], seconds[i] };
}

fragment XrbDualFloat xrb_dual_float(XrbDualVertex input [[stage_in]],
        const device float4 *colors [[buffer(0)]], constant uint &width [[buffer(1)]],
        const device float4 *seconds [[buffer(2)]]) {
    uint i = texel(input, width);
    return XrbDualFloat { colors[i], seconds[i] };
}
