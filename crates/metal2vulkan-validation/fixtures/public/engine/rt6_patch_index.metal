// Owned fixture: patch draws with patch and control-point index buffers, the questions Metal probes
// asked on an M4, with the domain points stored rather than rasterized. Written for this
// repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6ControlPoint {
    uint tag [[attribute(0)]];
};

struct Rt6PatchIn {
    patch_control_point<Rt6ControlPoint> points;
    uint ptag [[attribute(1)]];
};

// Every domain point of a patch stores the patch's word at the mark its first control point's
// tag names (`tag / 3`): `ptag << 24 | patch_id << 16 | tag0 << 8 | tag1 << 4 | tag2`, the
// per-patch attribute, the patch index, and the three control points' tags. Each patch's points
// store the same word, so the marks hold one word per patch drawn, whatever the invocation count
// and order; a culled patch stores nothing.
[[patch(triangle, 3)]]
vertex void rt6_patch_index(Rt6PatchIn in [[stage_in]],
                            uint patch [[patch_id]],
                            device uint *marks [[buffer(2)]]) {
    uint t0 = in.points[0].tag;
    uint t1 = in.points[1].tag;
    uint t2 = in.points[2].tag;
    marks[t0 / 3u] = (in.ptag << 24) | (patch << 16) | (t0 << 8) | (t1 << 4) | t2;
}
