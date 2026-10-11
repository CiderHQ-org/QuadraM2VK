#include <metal_stdlib>
using namespace metal;

struct VertexTriangleInput {
    float2 position [[attribute(0)]];
    float2 uv [[attribute(1)]];
};

struct VertexTriangleOutput {
    float4 position [[position]];
    float2 uv;
};

vertex VertexTriangleOutput vertex_varying_triangle(VertexTriangleInput input [[stage_in]]) {
    VertexTriangleOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}
