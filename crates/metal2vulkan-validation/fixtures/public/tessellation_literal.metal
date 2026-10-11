#include <metal_stdlib>
using namespace metal;

struct ControlPoint {
    float3 position [[attribute(0)]];
};

struct PatchInput {
    patch_control_point<ControlPoint> control_points;
    float4 color [[attribute(4)]];
};

struct RasterData {
    float4 position [[position]];
    float4 color [[user(locn0)]];
};

[[patch(quad, 16)]]
vertex RasterData tessellation_literal(
    PatchInput patch [[stage_in]],
    float2 coordinate [[position_in_patch]])
{
    RasterData output;
    output.position = float4(
        coordinate * 2.0 - 1.0 + patch.control_points[0].position.xy,
        0.0,
        1.0);
    output.color = patch.color;
    return output;
}
