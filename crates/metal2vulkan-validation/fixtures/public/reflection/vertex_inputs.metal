// Owned fixture: vertex functions whose stage inputs decide
// whether Metal's pipeline reflection calls a vertex descriptor's buffer used: an attribute the
// function declares but never reads (`air.arg_unused`), and one a function constant gates.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool reflection_use_uv [[function_constant(0)]];

struct ReflectionUnreadIn {
    float4 pos [[attribute(0)]];
    float2 uv [[attribute(1)]];
};

vertex float4 reflection_unread(ReflectionUnreadIn in [[stage_in]]) { return in.pos; }

struct ReflectionGatedIn {
    float4 pos [[attribute(0)]];
    float2 uv [[attribute(1), function_constant(reflection_use_uv)]];
};

vertex float4 reflection_gated(ReflectionGatedIn in [[stage_in]]) {
    float4 p = in.pos;
    if (reflection_use_uv) {
        p.xy += in.uv;
    }
    return p;
}

fragment float4 reflection_solid() { return float4(1.0); }
