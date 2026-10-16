// Owned fixture: control-point and patch-input fetches past their buffers,
// the control point containment form. A rasterization-disabled post-tessellation vertex function
// of three control points per triangle patch; every invocation of patch p writes the patch's
// three control-point words and its patch-input word to out[4p .. 4p + 4] (each store is the same
// at every domain point of the patch, so the result does not depend on how often the function
// runs). The case draws four patches over buffers that hold fewer elements, so the last patches
// fetch past both. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ControlPoint {
    uint c [[attribute(0)]];
};

struct PatchIn {
    patch_control_point<ControlPoint> cps;
    uint p [[attribute(1)]];
};

[[patch(triangle, 3)]]
vertex void ob3_control_point(PatchIn in [[stage_in]],
                              float3 bc [[position_in_patch]],
                              uint patch [[patch_id]],
                              device uint *out [[buffer(2)]]) {
    out[patch * 4u + 0u] = in.cps[0].c;
    out[patch * 4u + 1u] = in.cps[1].c;
    out[patch * 4u + 2u] = in.cps[2].c;
    out[patch * 4u + 3u] = in.p;
}
