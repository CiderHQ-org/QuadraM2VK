// Owned fixture: the stage-IO tail.
// - `bool` varyings, scalar and vector, written by a vertex function and read by a fragment
//   function: the interface carries each as a flat `uint` 0 or 1;
// - a fragment `[[primitive_id]]`, declared through whichever of `Geometry` or `Tessellation` the
//   profile enables;
// - fragment `[[render_target_array_index]]` and `[[viewport_array_index]]` inputs, through
//   `Geometry` or `ShaderLayer`, and `MultiViewport` or `ShaderViewportIndex`;
// - `[[barycentric_coord]]` as `float3` (perspective), `float2` (no perspective), and `float`
//.
// The vertex function reads a constant buffer, so the typed-pointer companion has a pointee.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Flagged {
    float4 position [[position]];
    bool lit;
    bool2 mask;
    float4 color;
};

vertex Flagged tail_vertex_bool(uint vid [[vertex_id]], constant float4 &scale [[buffer(0)]]) {
    Flagged out;
    out.position = float4(float(vid & 1u), float(vid >> 1), 0.0, 1.0);
    out.lit = (vid & 2u) != 0u;
    out.mask = bool2((vid & 1u) != 0u, vid > 1u);
    out.color = float4(float(vid), 0.25, 0.5, 1.0) * scale;
    return out;
}

fragment float4 tail_fragment_bool(Flagged in [[stage_in]]) {
    float4 c = in.lit ? in.color : float4(0.0, 0.0, 0.0, 1.0);
    if (in.mask.x) {
        c.x += 0.5;
    }
    if (in.mask.y) {
        c.y += 0.5;
    }
    return c;
}

fragment float4 tail_fragment_primitive(uint prim [[primitive_id]]) {
    return float4(float(prim & 7u), float(prim >> 3), 0.0, 1.0);
}

fragment float4 tail_fragment_layer(uint layer [[render_target_array_index]],
                                    uint viewport [[viewport_array_index]]) {
    return float4(float(layer), float(viewport), 0.0, 1.0);
}

fragment float4 tail_fragment_barycentric(float3 bary [[barycentric_coord]],
                                          float2 linear [[barycentric_coord, center_no_perspective]]) {
    return float4(bary.x, bary.y + bary.z, linear.x, linear.y);
}

fragment float4 tail_fragment_barycentric_scalar(float first [[barycentric_coord]]) {
    return float4(first, 0.0, 0.0, 1.0);
}
