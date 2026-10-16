// Owned fixture: vertex amplification into a layered render pass, the functions a Metal probe drew
// on an M4, so each lane pass of an amplified draw asks the question the probe measured. Written
// for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6Amplified {
    float4 position [[position]];
    uint tag [[flat]];
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth 0.5, once per instance and
// amplification: instance `i` tags its fragments `i + 1`, and the view mapping chooses the slice.
vertex Rt6Amplified rt6_amplified_vertex(uint vid [[vertex_id]], uint iid [[instance_id]]) {
    float2 a = float2(float((vid << 1) & 2u), float(vid & 2u));
    Rt6Amplified out;
    out.position = float4(a * 2.0 - 1.0, 0.5, 1.0);
    out.tag = iid + 1u;
    return out;
}

// The instance's tag above the amplification the fragment reads: `tag << 8 | amplification`.
fragment uint rt6_amplified_fragment(Rt6Amplified in [[stage_in]],
                                     ushort amp [[amplification_id]]) {
    return (in.tag << 8) | uint(amp);
}

// The same with the slice the fragment reads: `tag << 8 | amplification << 4 | slice`.
fragment uint rt6_amplified_slice_fragment(Rt6Amplified in [[stage_in]],
                                           ushort amp [[amplification_id]],
                                           uint slice [[render_target_array_index]]) {
    return (in.tag << 8) | (uint(amp) << 4) | slice;
}

struct Rt6AmplifiedLayered {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
    uint tag [[flat]];
};

// The fullscreen triangle into slice `layers[i]` for instance `i`, as `rt6_layered_vertex` draws
// it, under amplification.
vertex Rt6AmplifiedLayered rt6_amplified_layered_vertex(uint vid [[vertex_id]],
                                                        uint iid [[instance_id]],
                                                        constant uint *layers [[buffer(0)]]) {
    float2 a = float2(float((vid << 1) & 2u), float(vid & 2u));
    Rt6AmplifiedLayered out;
    out.position = float4(a * 2.0 - 1.0, 0.5, 1.0);
    out.layer = layers[iid];
    out.tag = iid + 1u;
    return out;
}

// The instance's tag, the amplification, and the slice the fragment reads:
// `tag << 8 | amplification << 4 | slice`.
fragment uint rt6_amplified_layered_fragment(Rt6AmplifiedLayered in [[stage_in]],
                                             ushort amp [[amplification_id]]) {
    return (in.tag << 8) | (uint(amp) << 4) | in.layer;
}
