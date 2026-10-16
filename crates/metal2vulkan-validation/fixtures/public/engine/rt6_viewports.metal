// Owned fixture: a vertex function that selects its viewport, the one a Metal probe drew (an M4),
// so each lane draw asks the question the probe measured. It adds the fragment function another
// probe drew, which reads the index. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6Depth {
    float z;
    float slope;
};

struct Rt6Selected {
    float4 position [[position]];
    uint viewport [[viewport_array_index]];
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth `z`, drawn through the
// viewport of its instance: `[[viewport_array_index]]` is `[[instance_id]]`, base instance
// included.
vertex Rt6Selected rt6_select(uint vid [[vertex_id]], uint iid [[instance_id]],
                              constant Rt6Depth &depth [[buffer(0)]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    Rt6Selected out;
    out.position = float4(4.0 * a - 1.0, depth.z, 1.0);
    out.viewport = iid;
    return out;
}

fragment float4 rt6_white() {
    return float4(1.0);
}

// The viewport index the fragment reads, in red (mod 256): Metal's fragment function reads the
// index its vertex function wrote, past the viewports set too (probe 7c).
fragment float4 rt6_viewport_read(Rt6Selected in [[stage_in]]) {
    return float4(float(in.viewport & 255u) / 255.0, 1.0, 0.0, 1.0);
}
