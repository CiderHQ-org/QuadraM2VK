// Owned fixture: the buffer and threadgroup arguments whose
// reflection Metal answers (MTLBufferBinding, MTLThreadgroupBinding): struct, vector, matrix,
// packed, atomic and byte pointees, constant and device address spaces, an unused argument, an
// argument a function constant gates, and arguments declared out of index order. Written for this
// repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool reflection_has_extra [[function_constant(0)]];
constant int reflection_scale [[function_constant(1)]];
constant float reflection_offset [[function_constant(2)]];

struct ReflectionPlain {
    float4 a;
    float b;
    packed_float3 c;
    int d[3];
    float3x3 m;
};

struct ReflectionInner {
    uint x;
    float2 y;
};

struct ReflectionNested {
    ReflectionInner in;
    ReflectionInner arr[2];
};

kernel void reflection_buffers(device ReflectionPlain *plain [[buffer(0)]],
                               constant float4 &uniform [[buffer(1)]],
                               device const uint *ro [[buffer(2)]],
                               device float3 *f3 [[buffer(3)]],
                               device ReflectionNested *nested [[buffer(4)]],
                               device uint *unused_arg [[buffer(5)]],
                               device uint *extra [[buffer(6), function_constant(reflection_has_extra)]],
                               device atomic_uint *counter [[buffer(7)]],
                               constant float4x4 &mat [[buffer(8)]],
                               device char *bytes [[buffer(9)]],
                               device float *written_only [[buffer(10)]],
                               uint tid [[thread_position_in_grid]]) {
    plain[tid].a += uniform * float(ro[tid]);
    f3[tid] += float3(nested[0].in.y, 1.0);
    if (reflection_has_extra) {
        extra[tid] = uint(reflection_scale);
    }
    float b = is_function_constant_defined(reflection_offset) ? reflection_offset : 0.0;
    atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    bytes[tid] = char(mat[0][0] + b);
    written_only[tid] = 1.0;
}

kernel void reflection_order(device uint *b3 [[buffer(3)]],
                             texture2d<float> t1 [[texture(1)]],
                             device uint *b0 [[buffer(0)]],
                             sampler s2 [[sampler(2)]],
                             texture2d<float> t0 [[texture(0)]],
                             sampler s0 [[sampler(0)]],
                             uint tid [[thread_position_in_grid]]) {
    float4 v = t1.sample(s2, float2(0.5), level(0)) + t0.sample(s0, float2(0.5), level(0));
    b3[tid] = uint(v.x);
    b0[tid] = uint(v.y);
}

kernel void reflection_threadgroup(threadgroup float4 *tg [[threadgroup(0)]],
                                   threadgroup uint *tg2 [[threadgroup(1)]],
                                   threadgroup ReflectionInner *tgs [[threadgroup(2)]],
                                   threadgroup float *tg_unused [[threadgroup(3)]],
                                   device float4 *out [[buffer(0)]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    tg[tid] = float4(float(tid));
    tg2[tid] = tid;
    if (tid == 0) {
        tgs[0].x = tid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[tid] = tg[0] + float(tg2[1]) + float(tgs[0].x);
}
