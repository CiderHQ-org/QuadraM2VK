// Owned fixture: a vertex attribute a function constant gates, read without
// a guard into a varying. MSL 5.8.1.3: with `has_tint` false (or undefined) the element is not
// declared in the input assembly structure, so the pipeline fetches nothing for it and the
// language gives its read no value; the engine reads zero and declares no attribute 1. With
// `has_tint` true it is an ordinary attribute. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool has_tint [[function_constant(0)]];

struct FcaAttributes {
    float4 position [[attribute(0)]];
    float4 tint [[attribute(1), function_constant(has_tint)]];
};

struct FcaVertexOut {
    float4 position [[position]];
    float4 tint;
};

vertex FcaVertexOut fca_vertex_absent_attribute(FcaAttributes in [[stage_in]]) {
    FcaVertexOut out;
    out.position = in.position;
    out.tint = in.tint;
    return out;
}
