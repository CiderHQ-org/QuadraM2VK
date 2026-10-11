#include <metal_stdlib>
using namespace metal;

constexpr sampler nearest_edge(coord::normalized, address::clamp_to_edge, filter::nearest);

// A resource handle chosen at runtime, then asked four different questions. An OpSelect on an
// image value is not valid SPIR-V, so every pure use has to be duplicated per arm and the RESULTS
// selected; the two textures differ in BOTH extent and content so each answer names its arm.
kernel void texture_select_queries(texture2d<float, access::sample> a [[texture(0)]],
                                   texture2d<float, access::sample> b [[texture(1)]],
                                   device float *out [[buffer(0)]],
                                   constant uint &pick [[buffer(1)]],
                                   uint gid [[thread_position_in_grid]]) {
    texture2d<float, access::sample> t = ((gid & 1u) == pick) ? a : b;
    out[gid * 4u + 0u] = float(t.get_width());
    out[gid * 4u + 1u] = float(t.get_height());
    out[gid * 4u + 2u] = t.read(uint2(1u, 0u)).x;
    out[gid * 4u + 3u] = t.sample(nearest_edge, float2(0.8f, 0.3f)).x;
}
