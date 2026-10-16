// Owned fixture: a vertex function that reads a stage
// input only where a function constant says, so a function made with the constant false leaves
// the attribute unread, which no earlier probe measured (MTLAttribute.isActive, and whether a
// pipeline calls the attribute's buffer used). Written for this repository; no third-party
// source. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool reflection_read_uv [[function_constant(0)]];

struct ReflectionBranchIn {
    float4 pos [[attribute(0)]];
    float2 uv [[attribute(1)]];
};

vertex float4 reflection_branch_read(ReflectionBranchIn in [[stage_in]]) {
    float4 p = in.pos;
    if (reflection_read_uv) {
        p.xy += in.uv;
    }
    return p;
}

fragment float4 reflection_branch_solid() { return float4(1.0); }
