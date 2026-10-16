// Owned fixture: a kernel's stage inputs, a post-tessellation vertex
// function's control-point and per-patch inputs, and argument bindings of each kind (a buffer, a
// texture, a sampler, threadgroup memory, an argument buffer, a constant reference), each read
// only where a function constant says, so a function made with the constant false leaves them
// unread: what Metal answers for their MTLAttribute.isActive and for a pipeline's binding `used`,
// which no earlier probe measured (specialized_reads measured a vertex function's vertex inputs).
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool reflection_read_more [[function_constant(0)]];

struct ReflectionGatedKernelIn {
    float4 a [[attribute(0)]];
    uint b [[attribute(1)]];
};

kernel void reflection_gated_stage_in(ReflectionGatedKernelIn in [[stage_in]],
                                      device float4 *out [[buffer(0)]],
                                      const device float *extra [[buffer(3)]],
                                      texture2d<float, access::read> tex [[texture(1)]],
                                      uint tid [[thread_position_in_grid]]) {
    float4 v = in.a;
    if (reflection_read_more) {
        v += float(in.b) + extra[tid] + tex.read(uint2(tid, 0));
    }
    out[tid] = v;
}

struct ReflectionGatedControlPoint {
    float4 p [[attribute(0)]];
    float4 q [[attribute(1)]];
};

struct ReflectionGatedPatchIn {
    patch_control_point<ReflectionGatedControlPoint> cps;
    float4 pf [[attribute(3)]];
    float4 pg [[attribute(4)]];
};

[[patch(quad, 4)]]
vertex float4 reflection_gated_tessellation(ReflectionGatedPatchIn in [[stage_in]],
                                            float2 uv [[position_in_patch]],
                                            constant float4 &tint [[buffer(5)]]) {
    float4 p = in.cps[0].p * uv.x + in.pf;
    if (reflection_read_more) {
        p += in.cps[1].q + in.pg + tint;
    }
    return p;
}

struct ReflectionGatedPointsIn {
    patch_control_point<ReflectionGatedControlPoint> cps;
    float4 pf [[attribute(3)]];
};

[[patch(quad, 4)]]
vertex float4 reflection_gated_control_points(ReflectionGatedPointsIn in [[stage_in]],
                                              float2 uv [[position_in_patch]]) {
    float4 p = in.pf * uv.x;
    if (reflection_read_more) {
        p += in.cps[2].q;
    }
    return p;
}

struct ReflectionGatedArgs {
    texture2d<float> t [[id(0)]];
    device float *d [[id(1)]];
};

kernel void reflection_gated_resources(device float4 *out [[buffer(0)]],
                                       constant ReflectionGatedArgs &args [[buffer(1)]],
                                       texture2d<float> tex [[texture(0)]],
                                       sampler smp [[sampler(0)]],
                                       threadgroup float *tg [[threadgroup(0)]],
                                       uint tid [[thread_position_in_grid]],
                                       uint lid [[thread_index_in_threadgroup]]) {
    float4 v = float4(1.0);
    if (reflection_read_more) {
        tg[lid] = float(tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v += tex.sample(smp, float2(0.5), level(0.0)) + args.t.read(uint2(0, 0)) + args.d[tid] +
             tg[lid ^ 1u];
    }
    out[tid] = v;
}

fragment float4 reflection_gated_solid() { return float4(1.0); }
