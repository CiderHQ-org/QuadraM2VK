// Owned fixture: the MTLFunction properties Metal answers from a
// metallib's function tags. Function constants a function reads only through
// is_function_constant_defined, only as an argument's predicate, for their value, or not at all;
// post-tessellation vertex functions of both patch types with several control-point counts; and
// vertex and kernel stage-input attributes declared out of index order, up to the highest index.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool kinds_defined_only [[function_constant(0)]];
constant bool kinds_gate [[function_constant(1)]];
constant int kinds_value [[function_constant(2)]];
constant float kinds_optional [[function_constant(3)]];
constant uint kinds_unused [[function_constant(4)]];

kernel void kinds_constants(device float *out [[buffer(0)]],
                            device float *gated [[buffer(1), function_constant(kinds_gate)]],
                            uint tid [[thread_position_in_grid]]) {
    out[tid] = is_function_constant_defined(kinds_defined_only) ? 1.0 : 0.0;
}

kernel void kinds_values(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float o = is_function_constant_defined(kinds_optional) ? kinds_optional : 2.0;
    out[tid] = kinds_value + int(o);
}

kernel void kinds_none(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}

struct KindsPoint {
    float4 p [[attribute(2)]];
    float2 q [[attribute(9)]];
};

struct KindsTriangle {
    patch_control_point<KindsPoint> cps;
    float4 pf [[attribute(17)]];
};

[[patch(triangle, 3)]]
vertex float4 kinds_triangle(KindsTriangle in [[stage_in]],
                             float3 bc [[position_in_patch]]) {
    return in.cps[0].p * bc.x + float4(in.cps[1].q, 0.0, 1.0) * bc.y + in.pf * bc.z;
}

struct KindsCorner {
    float4 p [[attribute(0)]];
};

struct KindsCorners {
    patch_control_point<KindsCorner> cps;
};

[[patch(quad, 16)]]
vertex float4 kinds_quad16(KindsCorners in [[stage_in]], float2 uv [[position_in_patch]]) {
    return in.cps[15].p * uv.x + in.cps[0].p * uv.y;
}

[[patch(triangle, 32)]]
vertex float4 kinds_triangle32(KindsCorners in [[stage_in]], float3 bc [[position_in_patch]]) {
    return in.cps[31].p * bc.x + in.cps[0].p * bc.y;
}

struct KindsOrder {
    float2 b [[attribute(5)]];
    float4 a [[attribute(1)]];
    uint c [[attribute(30)]];
};

vertex float4 kinds_order(KindsOrder in [[stage_in]]) {
    return in.a + float4(in.b, float(in.c), 0.0);
}

struct KindsKernelIn {
    uint c [[attribute(7)]];
    half2 h [[attribute(3)]];
};

kernel void kinds_stage_in(KindsKernelIn in [[stage_in]],
                           device float *out [[buffer(0)]],
                           uint tid [[thread_position_in_grid]]) {
    out[tid] = float(in.c) + float(in.h.x);
}
