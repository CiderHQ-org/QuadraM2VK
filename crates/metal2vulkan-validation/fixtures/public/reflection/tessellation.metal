// Owned fixture: a post-tessellation vertex function, whose
// MTLFunction reflection Metal answers with its patch type and control-point count and its
// control-point and per-patch attributes. Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ReflectionControlPoint {
    float4 p [[attribute(0)]];
};

struct ReflectionPatchIn {
    patch_control_point<ReflectionControlPoint> cps;
    float4 pf [[attribute(3)]];
};

[[patch(quad, 4)]]
vertex float4 reflection_tessellation(ReflectionPatchIn in [[stage_in]],
                                      float2 uv [[position_in_patch]]) {
    return in.cps[0].p * uv.x + in.pf;
}
