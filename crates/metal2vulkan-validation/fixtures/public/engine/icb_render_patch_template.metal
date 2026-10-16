// Owned fixture: the pipeline template of executed patch commands, read
// without rasterizing. A post-tessellation vertex function of triangle patches of one control
// point and no stage input returns nothing (its case runs with rasterization disabled) and
// stores, for every domain point it is called with, a mark at the point's cell of its patch's and
// instance's grid: `marks[(min(instance, 7) * 8 + min(patch, 7)) * 25 + j * 5 + i] =
// 0x10000 | instance << 8 | patch` (the low eight bits of each), where `i = rint(u * 4)` and
// `j = rint(v * 4)` for the point's position `(u, v, w)`. A domain point is shaded at least once and
// possibly more, and every store is idempotent, so a grid's marks are the set of the tessellator's
// points of that patch and instance, whatever the invocation count and order: factors of 1 mark
// the three corners (cells 0, 4 and 20), factors of 2 the corners, the edge midpoints (cells 2, 10
// and 12) and the centre (cell 6). Every operation is one rounding at most. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

[[patch(triangle, 1)]]
vertex void icb_render_patch_template(uint patch [[patch_id]],
                                      uint instance [[instance_id]],
                                      float3 uvw [[position_in_patch]],
                                      device uint *marks [[buffer(0)]])
{
    uint i = uint(clamp(rint(uvw.x * 4.0f), 0.0f, 4.0f));
    uint j = uint(clamp(rint(uvw.y * 4.0f), 0.0f, 4.0f));
    uint grid = min(instance, 7u) * 8u + min(patch, 7u);
    marks[grid * 25u + j * 5u + i] = 0x10000u | ((instance & 0xffu) << 8) | (patch & 0xffu);
}
