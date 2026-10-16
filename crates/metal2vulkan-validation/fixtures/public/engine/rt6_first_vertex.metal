// Owned fixture: a vertex function whose every vertex stores its own render target slice and
// viewport index, so the vertices of one primitive may store different ones, and a fragment
// function that reads both: which vertex's indices a primitive takes, asked of every primitive
// type. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6FirstVertex {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
    uint viewport [[viewport_array_index]];
};

// Vertex `vid` at `positions[vid]`, storing slice `indices[vid] & 0xff` and viewport
// `indices[vid] >> 8`.
vertex Rt6FirstVertex rt6_first_vertex(uint vid [[vertex_id]],
                                       const device float4 *positions [[buffer(0)]],
                                       const device uint *indices [[buffer(1)]]) {
    Rt6FirstVertex out;
    out.position = positions[vid];
    out.layer = indices[vid] & 0xffu;
    out.viewport = indices[vid] >> 8;
    return out;
}

struct Rt6FirstVertexPoint {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
    uint viewport [[viewport_array_index]];
    float size [[point_size]];
};

// `rt6_first_vertex` for points, 3 pixels wide: Metal refuses a point size outside the point class.
vertex Rt6FirstVertexPoint rt6_first_vertex_point(uint vid [[vertex_id]],
                                                  const device float4 *positions [[buffer(0)]],
                                                  const device uint *indices [[buffer(1)]]) {
    Rt6FirstVertexPoint out;
    out.position = positions[vid];
    out.layer = indices[vid] & 0xffu;
    out.viewport = indices[vid] >> 8;
    out.size = 3.0;
    return out;
}

// The viewport and slice the fragment reads: `0x10000 | viewport << 8 | slice`.
fragment uint rt6_first_vertex_read(uint layer [[render_target_array_index]],
                                    uint viewport [[viewport_array_index]]) {
    return 0x10000u | (viewport << 8) | layer;
}
