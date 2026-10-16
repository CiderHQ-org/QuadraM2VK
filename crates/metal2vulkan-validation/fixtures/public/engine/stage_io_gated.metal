// Owned fixture: a vertex attribute gated by a function constant
// and read without a guard. With `has_tint = false` the attribute is not declared and the
// language gives its read no value, so the engine reads zero and declares no attribute 1;
// with `has_tint = true` the attribute is an ordinary input. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool has_tint [[function_constant(0)]];

struct GatedAttributes {
    float4 position [[attribute(0)]];
    float4 tint [[attribute(1), function_constant(has_tint)]];
};

vertex float4 stage_io_vertex_gated(GatedAttributes in [[stage_in]],
                                     constant float4 &bias [[buffer(0)]]) {
    return in.position + in.tint + bias;
}
