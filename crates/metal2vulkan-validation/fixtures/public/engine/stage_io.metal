// Owned fixture: vertex fetch, varyings, stage builtins, fragment inputs
// and outputs. The vertex and fragment entries share `Varyings`, so their user keys match and
// the pair links (vertex outputs to fragment inputs by AIR user key). Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Attributes {
    float3 position [[attribute(0)]];
    half2 uv [[attribute(1)]];
    ushort2 ids [[attribute(4)]];
    uchar4 tint [[attribute(7)]];
};

struct Varyings {
    float4 position [[position]];
    half2 uv;
    float4 tint [[centroid_no_perspective]];
    uint id [[flat]];
    float point_size [[point_size]];
};

// Attributes of four widths, vertex and instance index, a constant buffer read in the vertex
// stage, a narrow varying, a flat integer, and an explicit point size.
vertex Varyings stage_io_vertex(Attributes in [[stage_in]],
                                constant float4 &scale [[buffer(1)]],
                                uint vid [[vertex_id]],
                                uint iid [[instance_id]]) {
    Varyings out;
    out.position = float4(in.position, 1.0) * scale;
    out.uv = in.uv * half(0.5);
    out.tint = float4(in.tint) / 255.0;
    out.id = vid * 7u + iid + uint(in.ids.x) + (uint(in.ids.y) << 16);
    out.point_size = 3.0;
    return out;
}

// A bare position return: one output that is not a struct; the module writes PointSize = 1.
vertex float4 stage_io_vertex_position(uint vid [[vertex_id]]) {
    return float4(float(vid & 1u), float(vid >> 1), 0.0, 1.0);
}

struct Clipped {
    float4 position [[position]];
    float clip [[clip_distance]] [2];
};

// A two-element clip distance array (shaderClipDistance).
vertex Clipped stage_io_vertex_clip(uint vid [[vertex_id]]) {
    Clipped out;
    out.position = float4(float(vid), 0.0, 0.0, 1.0);
    out.clip[0] = float(vid) - 1.0;
    out.clip[1] = 2.0 - float(vid);
    return out;
}

struct Layered {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
};

// Layered rendering from the vertex stage (shaderOutputLayer).
vertex Layered stage_io_vertex_layer(uint iid [[instance_id]]) {
    Layered out;
    out.position = float4(0.0, 0.0, 0.0, 1.0);
    out.layer = iid;
    return out;
}

struct Invariant {
    float4 position [[position, invariant]];
};

// An invariant position: `Position` decorated `Invariant`.
vertex Invariant stage_io_vertex_invariant(uint vid [[vertex_id]]) {
    Invariant out;
    out.position = float4(float(vid), 0.0, 0.0, 1.0);
    return out;
}

struct Targets {
    float4 color [[color(0)]];
    uint4 ids [[color(2)]];
    float depth [[depth(greater)]];
    uint mask [[sample_mask]];
};

// Varyings of every interpolation the vertex stage writes, front facing and the input sample
// mask; two color attachments, a greater-only depth, and an output sample mask.
fragment Targets stage_io_fragment(Varyings in [[stage_in]],
                                   bool front [[front_facing]],
                                   uint coverage [[sample_mask]]) {
    Targets out;
    out.color = float4(float2(in.uv), in.tint.z, front ? 1.0 : 0.5);
    out.ids = uint4(in.id, coverage, uint(in.position.x), 9u);
    out.depth = in.position.z * 0.5 + 0.5;
    out.mask = coverage & 0x5u;
    return out;
}

// Point coordinates, the sample index (sampleRateShading), and early fragment tests.
[[early_fragment_tests]]
fragment half4 stage_io_fragment_point(float2 pc [[point_coord]], uint sample [[sample_id]]) {
    return half4(half2(pc), half(sample), 1.0h);
}

// The primitive index (geometryShader on Vulkan).
fragment float4 stage_io_fragment_primitive(uint prim [[primitive_id]]) {
    return float4(float(prim), 0.0, 0.0, 1.0);
}

struct DualSource {
    float4 src0 [[color(0), index(0)]];
    float4 src1 [[color(0), index(1)]];
};

// Dual-source blending (dualSrcBlend).
fragment DualSource stage_io_fragment_dual_source(float4 position [[position]]) {
    DualSource out;
    out.src0 = float4(position.xy, 0.0, 1.0);
    out.src1 = float4(0.25);
    return out;
}
