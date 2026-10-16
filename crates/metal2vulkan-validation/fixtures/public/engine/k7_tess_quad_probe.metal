// Owned fixture: a quad patch's tessellation, read through its draw. Each
// patch covers the clip-space rectangle its per-patch attribute states (origin `rect.xy`, signed
// extent `rect.zw`, so a negative extent mirrors the patch), with `position_in_patch` mapped
// affinely onto it. `shade` (locn0) holds products of the domain coordinates: interpolated over
// the tessellated triangles, a product is exact only at their vertices, so a pixel's value pins
// where the tessellator put the vertices around it. `tag` (locn1) is the patch index, flat, so a
// pixel says which patch covers it, if any. Every operation is one rounding at most, and none
// can contract. Loop-free.
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

[[patch(quad, 1)]]
vertex Out k7_tess_quad_probe(PatchIn p [[stage_in]],
                              float2 uv [[position_in_patch]],
                              uint patch [[patch_id]])
{
    Out o;
    o.position = float4(p.rect.xy + uv * p.rect.zw, 0.0, 1.0);
    o.shade = float4(uv.x * uv.x, uv.y * uv.y, uv.x * uv.y, 1.0);
    o.tag = uint4(patch, 1, 2, 3);
    return o;
}
