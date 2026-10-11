// Owned fixture (WP-RT6, DESIGN D176, D305 item 22): a vertex function that chooses its render
// target slice and a fragment function that reads it, the pair RT6 Metal probe 12 drew (an M4), so
// each lane pass of a layered render pass asks the question the probe measured. Written for this
// repository; no third-party source. Loop-free.
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
