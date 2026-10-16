// Owned fixture: the functions the runtime's resolve lane draws with, the per-sample values of a
// Metal probe on an M4 (macOS 27.0.1), so each lane resolve asks the question the probe measured.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6ResolveOut {
    float4 position [[position]];
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth `z`.
vertex Rt6ResolveOut rt6_resolve_full(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    Rt6ResolveOut out;
    out.position = float4(4.0 * a - 1.0, z, 1.0);
    return out;
}

// Half the target, (-1, -1), (1, -1), (-1, 1): the pixels on its diagonal are half covered.
vertex Rt6ResolveOut rt6_resolve_half(uint vid [[vertex_id]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    Rt6ResolveOut out;
    out.position = float4(2.0 * a - 1.0, 0.5, 1.0);
    return out;
}

fragment float4 rt6_resolve_white() {
    return float4(1.0);
}

// Per sample: red s / 3, green 1 on sample 3 only, blue 0.5, alpha 1.
fragment float4 rt6_resolve_samples(uint s [[sample_id]]) {
    return float4(float(s) / 3.0, s == 3u ? 1.0 : 0.0, 0.5, 1.0);
}

// Per sample: s, 0.1 s, -s, 0.25 s.
fragment float4 rt6_resolve_sample_values(uint s [[sample_id]]) {
    return float4(float(s), 0.1 * float(s), -float(s), 0.25 * float(s));
}

struct Rt6ResolveMasked {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};

// White on the samples of `mask` only.
fragment Rt6ResolveMasked rt6_resolve_masked(constant uint &mask [[buffer(0)]]) {
    Rt6ResolveMasked out;
    out.color = float4(1.0);
    out.mask = mask;
    return out;
}
