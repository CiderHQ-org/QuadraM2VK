#include <metal_stdlib>
using namespace metal;

kernel void kernel_quad_ops_linear(
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_threadgroup]])
{
    float4 v = float4(float(tid), float(tid) * 2.0f, float(tid) * 4.0f, float(tid) * 8.0f);
    float4 s = quad_sum(v);
    int m = quad_max(int(tid) * 3 - 40);
    int n = quad_min(int(tid) * 3 - 40);
    float b = quad_broadcast(float(tid), 2u);
    float x = quad_shuffle_xor(float(tid), 3u);
    uint o = tid * 8;
    out[o + 0] = as_type<uint>(s.x);
    out[o + 1] = as_type<uint>(s.y);
    out[o + 2] = as_type<uint>(s.z);
    out[o + 3] = as_type<uint>(s.w);
    out[o + 4] = as_type<uint>(m);
    out[o + 5] = as_type<uint>(n);
    out[o + 6] = as_type<uint>(b);
    out[o + 7] = as_type<uint>(x);
}
