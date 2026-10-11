#include <metal_stdlib>
using namespace metal;

kernel void kernel_simd_shuffle_and_fill(
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint d = tid + 1;
    uint f = 100 + tid;
    uint su = simd_shuffle_and_fill_up(d, f, 3);
    uint sd = simd_shuffle_and_fill_down(d, f, 5);
    uint2 d2 = uint2(d, d * 2);
    uint2 q2 = simd_shuffle_down(d2, 7);

    float3 fd = float3(float(d), float(d) * 2.0f, float(d) * 4.0f);
    float3 ff = float3(float(f), float(f) * 2.0f, float(f) * 4.0f);
    float3 u3 = simd_shuffle_and_fill_up(fd, ff, 2);
    float3 v3 = simd_shuffle_and_fill_down(fd, ff, 4);
    float3 w3 = simd_shuffle_up(fd, 6);

    float ru = simd_shuffle_and_fill_up(float(d), float(f), 1);
    float rv = simd_shuffle_and_fill_down(float(d), float(f), 9);

    half4 hd = half4(half(d), half(d) + 1.0h, half(d) + 2.0h, half(d) + 3.0h);
    half4 hf = half4(half(f), half(f) + 1.0h, half(f) + 2.0h, half(f) + 3.0h);
    half4 h4 = simd_shuffle_and_fill_down(hd, hf, 3);
    half3 h3 = half3(half(d), half(d) * 2.0h, half(d) * 4.0h);
    half3 g3 = simd_shuffle_down(h3, 5);

    uint b = tid * 22;
    out[b + 0] = su;
    out[b + 1] = sd;
    out[b + 2] = q2.x;
    out[b + 3] = q2.y;
    out[b + 4] = as_type<uint>(u3.x);
    out[b + 5] = as_type<uint>(u3.y);
    out[b + 6] = as_type<uint>(u3.z);
    out[b + 7] = as_type<uint>(v3.x);
    out[b + 8] = as_type<uint>(v3.y);
    out[b + 9] = as_type<uint>(v3.z);
    out[b + 10] = as_type<uint>(w3.x);
    out[b + 11] = as_type<uint>(w3.y);
    out[b + 12] = as_type<uint>(w3.z);
    out[b + 13] = as_type<uint>(ru);
    out[b + 14] = as_type<uint>(rv);
    out[b + 15] = as_type<ushort>(h4.x);
    out[b + 16] = as_type<ushort>(h4.y);
    out[b + 17] = as_type<ushort>(h4.z);
    out[b + 18] = as_type<ushort>(h4.w);
    out[b + 19] = as_type<ushort>(g3.x);
    out[b + 20] = as_type<ushort>(g3.y);
    out[b + 21] = as_type<ushort>(g3.z);

    // The plain shuffles have no fill operand: Metal leaves the lanes whose source index
    // leaves the simdgroup undefined, so overwrite exactly those slots with a constant.
    if (tid < 6) {
        out[b + 10] = 0;
        out[b + 11] = 0;
        out[b + 12] = 0;
    }
    if (tid >= 25) {
        out[b + 2] = 0;
        out[b + 3] = 0;
    }
    if (tid >= 27) {
        out[b + 19] = 0;
        out[b + 20] = 0;
        out[b + 21] = 0;
    }
}
