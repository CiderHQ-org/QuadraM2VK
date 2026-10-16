// Owned fixture: the functions two Metal probes drew in a rate-mapped pass over a 2-slice array
// target, so each lane draw asks the question the probes measured. A fullscreen triangle, (-1, -1),
// (3, -1), (-1, 3); a vertex function that renders it into the slice of its instance; one that
// reads its amplification and one that reads
// nothing; and fragment functions that write their position, 1, and (where the vertex function
// chose one) the slice. Written for this repository; no third-party source. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct CfbRlVertex {
    float4 position [[position]];
};

struct CfbRlLayered {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
};

static float4 cfb_rl_corner(uint vid) {
    return float4(vid == 1u ? 3.0 : -1.0, vid == 2u ? 3.0 : -1.0, 0.0, 1.0);
}

// The triangle in slice `instance_id`.
vertex CfbRlLayered cfb_rl_layer(uint vid [[vertex_id]], uint instance [[instance_id]]) {
    CfbRlLayered out;
    out.position = cfb_rl_corner(vid);
    out.layer = instance;
    return out;
}

// The triangle, amplified: the amplification id moves nothing but its view mapping's slice.
vertex CfbRlVertex cfb_rl_amp(uint vid [[vertex_id]], ushort amplification [[amplification_id]]) {
    CfbRlVertex out;
    out.position = cfb_rl_corner(vid);
    return out;
}

vertex CfbRlVertex cfb_rl_full(uint vid [[vertex_id]]) {
    CfbRlVertex out;
    out.position = cfb_rl_corner(vid);
    return out;
}

fragment float4 cfb_rl_position(CfbRlVertex in [[stage_in]]) {
    return float4(in.position.xy, 1.0, 0.0);
}

fragment float4 cfb_rl_position_layer(CfbRlLayered in [[stage_in]]) {
    return float4(in.position.xy, 1.0, float(in.layer));
}
