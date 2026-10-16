// Owned fixture: a post-tessellation vertex
// function whose patch states no control-point count, as a raw module. It is
// metallib-only/function_types.metal's `types_quad_any`, word for word, whose metallib answers
// Metal recorded (`== functions_function_types` in metal-answers.txt): the raw module answers the
// same properties from its AIR. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct TypesCorner {
    float4 p [[attribute(0)]];
};

struct TypesCorners {
    patch_control_point<TypesCorner> cps;
};

[[patch(quad)]]
vertex float4 types_quad_any(TypesCorners in [[stage_in]], float2 uv [[position_in_patch]]) {
    return in.cps[0].p * uv.x;
}
