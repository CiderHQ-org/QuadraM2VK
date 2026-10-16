// Owned fixture: the functions a Metal probe drew on an M4 in a render pass without attachments at
// a default raster sample count of 1, 2 and 4, so each lane draw asks the question the probe
// measured. The fullscreen triangle covers a 2x2 render area; each fragment invocation ORs into its
// pixel's word the samples it ran for and counts itself in word 4. Metal runs both functions once
// per covered sample (each writes memory), so each pixel's word holds every sample and the count is
// the pixels times the samples. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6Point {
    float4 position [[position]];
};

// The fullscreen triangle, (-1, -1), (3, -1), (-1, 3), computed rather than indexed from a local
// array (whose lifetime markers LLVM majors print differently).
vertex Rt6Point rt6_cover(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return Rt6Point { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

// Sample rate: each invocation ORs in its own sample.
fragment void rt6_mark_sample(Rt6Point in [[stage_in]], uint sample [[sample_id]],
                              device atomic_uint *words [[buffer(0)]]) {
    uint pixel = uint(in.position.y) * 2u + uint(in.position.x);
    atomic_fetch_or_explicit(&words[pixel], 1u << sample, memory_order_relaxed);
    atomic_fetch_add_explicit(&words[4], 1u, memory_order_relaxed);
}

// The coverage mask: each invocation ORs in the samples it covers.
fragment void rt6_mark_coverage(Rt6Point in [[stage_in]], uint mask [[sample_mask]],
                                device atomic_uint *words [[buffer(0)]]) {
    uint pixel = uint(in.position.y) * 2u + uint(in.position.x);
    atomic_fetch_or_explicit(&words[pixel], mask, memory_order_relaxed);
    atomic_fetch_add_explicit(&words[4], 1u, memory_order_relaxed);
}
