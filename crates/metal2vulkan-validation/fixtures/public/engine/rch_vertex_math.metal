// Owned fixture: a vertex function of a typical vertex shader's
// arithmetic, for the cost of an invariant position: each vertex takes a position from buffer 0
// at its vertex id, transforms it by a model, a view and a projection matrix from buffer 1 (each
// a float4x4), and its normal by the model's 3x3 part, and writes the position, the transformed
// normal and a texture coordinate. The fragment function writes the normal as a color. Written
// for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct RchVertexMathOut {
    float4 position [[position]];
    float3 normal;
    float2 uv;
};

struct RchVertexMathUniforms {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
};

vertex RchVertexMathOut rch_vertex_math(uint vid [[vertex_id]],
                                        const device float4 *positions [[buffer(0)]],
                                        constant RchVertexMathUniforms &u [[buffer(1)]]) {
    float4 p = positions[vid];
    float4 world = u.model * p;
    RchVertexMathOut out;
    out.position = u.projection * (u.view * world);
    float3x3 normal_matrix = float3x3(u.model[0].xyz, u.model[1].xyz, u.model[2].xyz);
    out.normal = normalize(normal_matrix * p.xyz);
    out.uv = world.xy * 0.5 + 0.5;
    return out;
}

fragment float4 rch_vertex_math_color(RchVertexMathOut in [[stage_in]]) {
    return float4(in.normal * 0.5 + 0.5, 1.0);
}
