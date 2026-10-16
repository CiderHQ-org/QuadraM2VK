// Owned fixture: the functions a Metal probe drew under a depth compare of Never in a 4x4 pass
// without a depth attachment, so each lane draw asks the question the probe measured. A fullscreen
// triangle at depth 0.5; fragment functions that count their invocations in word 0 of a buffer and
// return red, one under [[early_fragment_tests]] and one that discards column 0 after counting; and
// a vertex function that counts its invocations in word 1. cfb_dn_red_early: red under
// [[early_fragment_tests]], writing no memory. Written for this repository; no third-party source.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct CfbDnVertex {
    float4 position [[position]];
};

// The fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth 0.5.
static CfbDnVertex cfb_dn_corner(uint vid) {
    CfbDnVertex out;
    out.position = float4(vid == 1u ? 3.0 : -1.0, vid == 2u ? 3.0 : -1.0, 0.5, 1.0);
    return out;
}

vertex CfbDnVertex cfb_dn_full(uint vid [[vertex_id]]) {
    return cfb_dn_corner(vid);
}

vertex CfbDnVertex cfb_dn_count_vertices(uint vid [[vertex_id]],
                                         device atomic_uint *c [[buffer(1)]]) {
    atomic_fetch_add_explicit(&c[1], 1u, memory_order_relaxed);
    return cfb_dn_corner(vid);
}

fragment float4 cfb_dn_count(CfbDnVertex in [[stage_in]], device atomic_uint *c [[buffer(0)]]) {
    atomic_fetch_add_explicit(&c[0], 1u, memory_order_relaxed);
    return float4(1.0, 0.0, 0.0, 1.0);
}

[[early_fragment_tests]]
fragment float4 cfb_dn_count_early(CfbDnVertex in [[stage_in]],
                                   device atomic_uint *c [[buffer(0)]]) {
    atomic_fetch_add_explicit(&c[0], 1u, memory_order_relaxed);
    return float4(1.0, 0.0, 0.0, 1.0);
}

fragment float4 cfb_dn_count_discard(CfbDnVertex in [[stage_in]],
                                     device atomic_uint *c [[buffer(0)]]) {
    atomic_fetch_add_explicit(&c[0], 1u, memory_order_relaxed);
    if (uint(in.position.x) == 0u) {
        discard_fragment();
    }
    return float4(1.0, 0.0, 0.0, 1.0);
}

[[early_fragment_tests]]
fragment float4 cfb_dn_red_early(CfbDnVertex in [[stage_in]]) {
    return float4(1.0, 0.0, 0.0, 1.0);
}
