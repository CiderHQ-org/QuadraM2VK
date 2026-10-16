// Owned fixture: vertex and fragment functions whose pipeline
// reflection Metal answers per stage: a vertex stage input that a vertex descriptor feeds (Metal
// lists the descriptor's buffers as `vertexBuffer.N`), buffers, textures and samplers in both
// stages, and a vertex function without a stage input. Written for this repository; no
// third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ReflectionVertexIn {
    float4 pos [[attribute(0)]];
    float2 uv [[attribute(1)]];
    half4 color [[attribute(2)]];
};

struct ReflectionVertexOut {
    float4 pos [[position]];
    float2 uv;
    half4 color;
};

vertex ReflectionVertexOut reflection_vertex(ReflectionVertexIn in [[stage_in]],
                                             constant float4x4 &mvp [[buffer(1)]],
                                             device const float *extra [[buffer(3)]],
                                             texture2d<float> vt [[texture(0)]],
                                             uint vid [[vertex_id]],
                                             uint iid [[instance_id]]) {
    constexpr sampler ls(filter::linear);
    ReflectionVertexOut o;
    o.pos = mvp * in.pos + vt.sample(ls, in.uv, level(0)) + extra[vid] + float(iid);
    o.uv = in.uv;
    o.color = in.color;
    return o;
}

fragment float4 reflection_fragment(ReflectionVertexOut in [[stage_in]],
                                    texture2d<float> t [[texture(0)]],
                                    sampler s [[sampler(0)]],
                                    constant float4 &tint [[buffer(0)]],
                                    device atomic_uint *count [[buffer(2)]]) {
    atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    return t.sample(s, in.uv) * tint + float4(in.color);
}

vertex float4 reflection_vertex_plain(uint vid [[vertex_id]], device const float4 *pos [[buffer(0)]]) {
    return pos[vid];
}

fragment float4 reflection_fragment_solid() { return float4(1.0); }
