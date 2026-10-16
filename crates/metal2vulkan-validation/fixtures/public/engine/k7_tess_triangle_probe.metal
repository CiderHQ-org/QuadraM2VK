// Owned fixture: a triangle patch's tessellation, read through its draw.
// Each patch covers the clip-space right triangle its per-patch attribute states: the domain's
// corner (1, 0, 0) at `rect.xy + (rect.z, 0)`, (0, 1, 0) at `rect.xy + (0, rect.w)`, and
// (0, 0, 1) at `rect.xy` (a negative extent mirrors the patch). `shade` (locn0) holds the squares
// of the barycentric domain coordinates: interpolated over the tessellated triangles, a square is
// exact only at their vertices, so a pixel's value pins where the tessellator put the vertices
// around it. `tag` (locn1) is the patch index, flat, so a pixel says which patch covers it, if
// any. Every operation is one rounding at most, and none can contract. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct PatchIn {
    float4 rect [[attribute(0)]];
};

struct Out {
    float4 position [[position]];
    float4 shade [[user(locn0)]];
    uint4 tag [[user(locn1)]];
};

[[patch(triangle, 1)]]
vertex Out k7_tess_triangle_probe(PatchIn p [[stage_in]],
                                  float3 bc [[position_in_patch]],
                                  uint patch [[patch_id]])
{
    Out o;
    o.position = float4(p.rect.xy + bc.xy * p.rect.zw, 0.0, 1.0);
    o.shade = float4(bc.x * bc.x, bc.y * bc.y, bc.z * bc.z, 1.0);
    o.tag = uint4(patch, 1, 2, 3);
    return o;
}
