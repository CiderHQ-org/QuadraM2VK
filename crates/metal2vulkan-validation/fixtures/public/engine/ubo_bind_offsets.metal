// Owned fixture: constant buffers read at constant offsets in each stage, for the runtime test
// that binds them at offsets the device's uniform descriptor alignment does and does not divide
// (`tests/uniform_bind_offsets.rs`): the read either takes the bind offset's residual or, where
// the range starts at the bind offset, the Metal offset alone. Written for this repository; no
// third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct UboVaryings {
    float4 position [[position]];
    float alpha [[flat]];
};

// A triangle that covers the target; its alpha is k[1].
vertex UboVaryings ubo_offsets_vertex(uint vid [[vertex_id]], constant float *k [[buffer(1)]]) {
    float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));
    UboVaryings out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.alpha = k[1];
    return out;
}

// The color c[0], c[1], c[2] with the vertex function's alpha.
fragment float4 ubo_offsets_fragment(UboVaryings in [[stage_in]], constant float *c [[buffer(0)]]) {
    return float4(c[0], c[1], c[2], in.alpha);
}

// c[0] and c[2] into out.
kernel void ubo_offsets_kernel(constant float *c [[buffer(0)]], device float *out [[buffer(1)]]) {
    out[0] = c[0];
    out[1] = c[2];
}
