// Owned fixture: the domain points of a quad patch, read without
// rasterizing them. The post-tessellation vertex function returns nothing (the case runs with
// rasterization disabled) and stores, for every domain point it is called with:
// - a mark at the point's cell, `marks[j * 65 + i] = 1`, where `i = rint(u * grid.x)` and
//   `j = rint(v * grid.y)`, clamped to [0, 64]. The per-patch attribute `grid` states the two
//   scales (its z and w are unused), so a case can keep every point away from a cell boundary.
// - the point's coordinates, bit for bit, at `marks[4225 + 2 * cell]` (u) and the word after (v).
// A domain point is shaded at least once and possibly more, and every store is idempotent, so the
// marks are the set of the tessellator's points whatever the invocation count and order. The
// segment count and spacing of each edge, and the inner rings, are where the marks are: which
// factor drives which edge, how each partition mode rounds and places a fractional factor, and
// what the inside factors do. Every operation is one rounding at most. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct PatchIn {
    float4 grid [[attribute(0)]];
};

[[patch(quad, 1)]]
vertex void k7_tess_quad_domain(PatchIn p [[stage_in]],
                                float2 uv [[position_in_patch]],
                                device uint* marks [[buffer(1)]])
{
    float fi = clamp(rint(uv.x * p.grid.x), 0.0f, 64.0f);
    float fj = clamp(rint(uv.y * p.grid.y), 0.0f, 64.0f);
    uint cell = uint(fj) * 65u + uint(fi);
    marks[cell] = 1u;
    marks[4225u + 2u * cell] = as_type<uint>(uv.x);
    marks[4226u + 2u * cell] = as_type<uint>(uv.y);
}
