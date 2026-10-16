// Owned fixture: a `float4` output computed by whole-vector arithmetic over
// loads under fast math, for the computed-class lane of the extended-range blend table
// (`metal2vulkan_devtools::xr_blends::computed_table`). The fullscreen triangle of `rt6_samples`;
// the fragment returns its texel's value times its texel's entry of a buffer of ones, each read
// at `values[y * width + x]`: the loaded value, computed.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct XrbComputedVertex {
    float4 position [[position]];
};

vertex XrbComputedVertex xrb_computed_vertex(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return XrbComputedVertex { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

fragment float4 xrb_computed_float(XrbComputedVertex input [[stage_in]],
        const device float4 *colors [[buffer(0)]], constant uint &width [[buffer(1)]],
        const device float4 *ones [[buffer(2)]]) {
    uint i = uint(input.position.y) * width + uint(input.position.x);
    return colors[i] * ones[i];
}
