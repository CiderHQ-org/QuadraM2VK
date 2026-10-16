// Owned fixture: a vertex function that places each vertex at the clip-space position buffer 0
// holds for it, and fragment functions that return the three weights of [[barycentric_coord]] as
// a float3, perspective-correct and not, so a lane draw compares each weight with Metal's. A
// second vertex function adds varyings (a smooth float4, a flat uint and a float2 the linker may
// pack beside it), which one fragment function reads with the weights and the primitive ID and
// another without the weights, so a lane compares what each pixel reads of them.
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct BwVertex {
    float4 position [[position]];
};

vertex BwVertex bw_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]]) {
    BwVertex out;
    out.position = positions[vid];
    return out;
}

fragment float4 bw_perspective(float3 weights [[barycentric_coord, center_perspective]]) {
    return float4(weights, 1.0);
}

fragment float4 bw_linear(float3 weights [[barycentric_coord, center_no_perspective]]) {
    return float4(weights, 1.0);
}

struct BvVertex {
    float4 position [[position]];
    float4 color;
    uint tag [[flat]];
    float2 uv;
};

struct BvOut {
    float4 weights [[color(0)]];
    float4 color [[color(1)]];
    uint4 rest [[color(2)]];
};

vertex BvVertex bv_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]]) {
    BvVertex out;
    float4 p = positions[vid];
    out.position = p;
    out.color = float4(p.x * 0.75, p.y * 0.5 + 0.25, float(vid) / 7.0, 1.0 / p.w);
    out.tag = vid * 7u + 3u;
    out.uv = p.yx * 0.375;
    return out;
}

fragment BvOut bv_weights(BvVertex in [[stage_in]],
                          float3 weights [[barycentric_coord, center_perspective]],
                          uint primitive [[primitive_id]]) {
    BvOut out;
    out.weights = float4(weights, 1.0);
    out.color = in.color;
    out.rest = uint4(in.tag, primitive, as_type<uint>(in.uv.x), as_type<uint>(in.uv.y));
    return out;
}

fragment BvOut bv_plain(BvVertex in [[stage_in]], uint primitive [[primitive_id]]) {
    BvOut out;
    out.weights = float4(0.0, 0.0, 0.0, 1.0);
    out.color = in.color;
    out.rest = uint4(in.tag, primitive, as_type<uint>(in.uv.x), as_type<uint>(in.uv.y));
    return out;
}
