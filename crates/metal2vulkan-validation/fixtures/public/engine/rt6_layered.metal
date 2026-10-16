// Owned fixture: a vertex function that chooses its render target slice and a fragment function
// that reads it, the pair a Metal probe drew on an M4, so each lane pass of a layered render pass
// asks the question the probe measured. Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6Layered {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
    uint tag [[flat]];
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth 0.5, once per instance:
// instance `i` renders into slice `layers[i]` of the pass and tags its fragments `i + 1`.
vertex Rt6Layered rt6_layered_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                     constant uint *layers [[buffer(0)]]) {
    float2 a = float2(float((vid << 1) & 2u), float(vid & 2u));
    Rt6Layered out;
    out.position = float4(a * 2.0 - 1.0, 0.5, 1.0);
    out.layer = layers[iid];
    out.tag = iid + 1u;
    return out;
}

// The instance's tag above the slice the fragment reads: `tag << 8 | layer`.
fragment uint rt6_layered_fragment(Rt6Layered in [[stage_in]]) {
    return (in.tag << 8) | in.layer;
}

// The slice and tag `rt6_layered_fragment` returns, with the primitive ID in bits 16 to 22 and, in
// bit 23, whether every barycentric weight lies in [0, 1]: the reads one synthesized geometry
// stage serves together on a device without the slice output or barycentrics.
fragment uint rt6_layered_weights(Rt6Layered in [[stage_in]],
                                  float3 weights [[barycentric_coord]],
                                  uint primitive [[primitive_id]]) {
    uint inside = (all(weights >= 0.0) && all(weights <= 1.0)) ? 1u : 0u;
    return (inside << 23) | ((primitive & 0x7fu) << 16) | (in.tag << 8) | in.layer;
}
