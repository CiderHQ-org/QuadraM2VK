// Post-tessellation vertex functions as Vulkan tessellation evaluation
// shaders. Owned MSL; see README.md for each entry's purpose.
#include <metal_stdlib>
using namespace metal;

struct Out {
    float4 position [[position]];
    float4 color [[user(locn0)]];
};

// A quad patch whose control points the function reads from a buffer by its patch index: no
// vertex descriptor is involved.
[[patch(quad, 4)]]
vertex Out tess_quad_buffer(float2 uv [[position_in_patch]],
                            uint patch [[patch_id]],
                            const device float4 *cps [[buffer(0)]])
{
    float4 a = mix(cps[patch * 4 + 0], cps[patch * 4 + 1], uv.x);
    float4 b = mix(cps[patch * 4 + 3], cps[patch * 4 + 2], uv.x);
    Out o;
    o.position = mix(a, b, uv.y);
    o.color = float4(uv, float(patch), 1.0);
    return o;
}

struct ControlPoint {
    float4 position [[attribute(0)]];
    half2 uv [[attribute(1)]];
    uchar4 color [[attribute(2)]];
};

struct PatchIn {
    patch_control_point<ControlPoint> cps;
    float4 tint [[attribute(5)]];
};

// A triangle patch through the vertex descriptor: control points (a field no instruction reads
// stays unfetched), a per-patch attribute, and the instance index.
[[patch(triangle, 3)]]
vertex Out tess_triangle_points(PatchIn p [[stage_in]],
                                float3 bc [[position_in_patch]],
                                uint instance [[instance_id]])
{
    Out o;
    o.position = p.cps[0].position * bc.x + p.cps[1].position * bc.y +
                 p.cps[2].position * bc.z;
    o.position.x += float(p.cps[2].uv.x);
    o.color = p.tint + float4(float(instance));
    return o;
}

// A control point read at an index the function computes.
[[patch(quad, 4)]]
vertex Out tess_dynamic_point(PatchIn p [[stage_in]],
                              float2 uv [[position_in_patch]],
                              constant uint &k [[buffer(1)]])
{
    Out o;
    o.position = p.cps[k].position * uv.x;
    o.color = float4(p.cps[k & 3].color) / 255.0;
    return o;
}

// A control point used whole (the compiler merges the two fetches' structs in a phi): its
// unused color field stays unfetched.
[[patch(triangle, 3)]]
vertex Out tess_whole_point(PatchIn p [[stage_in]],
                            float3 bc [[position_in_patch]],
                            constant uint &k [[buffer(1)]])
{
    ControlPoint c = k == 7 ? p.cps[1] : p.cps[2];
    Out o;
    o.position = c.position * bc.x;
    o.color = float4(float(c.uv.y));
    return o;
}

// `ushort` patch and instance indices: each holds the low 16 bits of the 32-bit index.
[[patch(quad, 4)]]
vertex Out tess_short_ids(float2 uv [[position_in_patch]],
                          ushort patch [[patch_id]],
                          ushort instance [[instance_id]])
{
    Out o;
    o.position = float4(uv, 0.0, 1.0);
    o.color = float4(float(patch), float(instance), 0.0, 1.0);
    return o;
}

struct SizedOut {
    float4 position [[position]];
    float size [[point_size]];
};

// A point size: the tessellator emits triangles, which no point size sizes, so the evaluation
// shader drops it (exact) rather than need the tessellation point-size feature for nothing.
[[patch(quad, 4)]]
vertex SizedOut tess_point_size(float2 uv [[position_in_patch]])
{
    SizedOut o;
    o.position = float4(uv, 0.0, 1.0);
    o.size = 2.0;
    return o;
}
