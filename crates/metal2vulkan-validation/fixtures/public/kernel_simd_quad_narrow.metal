#include <metal_stdlib>
using namespace metal;

kernel void kernel_simd_quad_narrow(
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint d = tid + 1;

    half4 hd = half4(half(d), half(d) + 1.0h, half(d) + 2.0h, half(d) + 3.0h);
    half4 h4 = simd_shuffle_down(hd, 5);

    uchar  ub = uchar(d * 3);
    char   sb = char(int(d) * 3 - 40);
    short  ss = short(int(d) * 300 - 5000);
    uchar  ru = simd_shuffle_down(ub, 3);
    char   rs = simd_shuffle_down(sb, 3);
    short  rh = simd_shuffle_down(ss, 6);

    float3 f3 = float3(float(d), float(d) * 2.0f, float(d) * 4.0f);
    float2 f2 = float2(float(d), float(d) * 8.0f);
    float3 s3 = quad_sum(f3);
    float2 s2 = quad_sum(f2);
    float3 q3 = quad_shuffle_down(f3, 1);
    float2 q2 = quad_shuffle_down(f2, 2);

    uint b = tid * 17;
    out[b + 0] = uint(as_type<ushort>(h4.x));
    out[b + 1] = uint(as_type<ushort>(h4.y));
    out[b + 2] = uint(as_type<ushort>(h4.z));
    out[b + 3] = uint(as_type<ushort>(h4.w));
    out[b + 4] = uint(ru);
    out[b + 5] = as_type<uint>(int(rs));
    out[b + 6] = as_type<uint>(int(rh));
    out[b + 7] = as_type<uint>(s3.x);
    out[b + 8] = as_type<uint>(s3.y);
    out[b + 9] = as_type<uint>(s3.z);
    out[b + 10] = as_type<uint>(s2.x);
    out[b + 11] = as_type<uint>(s2.y);
    out[b + 12] = as_type<uint>(q3.x);
    out[b + 13] = as_type<uint>(q3.y);
    out[b + 14] = as_type<uint>(q3.z);
    out[b + 15] = as_type<uint>(q2.x);
    out[b + 16] = as_type<uint>(q2.y);

    // Every shuffle here is a plain one with no fill operand, so Metal leaves the lane
    // undefined whenever the source index leaves the group -- the simdgroup for the simd_*
    // shuffles, the quad for the quad_* ones. Overwrite exactly those slots with a constant
    // rather than assert on undefined behaviour. The branch is on tid, never on the shuffle
    // result: a select on the result would let SPIRV-Cross sink the shuffle into a ternary arm.
    if (tid >= 27) {
        out[b + 0] = 0;
        out[b + 1] = 0;
        out[b + 2] = 0;
        out[b + 3] = 0;
    }
    if (tid >= 29) {
        out[b + 4] = 0;
        out[b + 5] = 0;
    }
    if (tid >= 26) {
        out[b + 6] = 0;
    }
    if (tid % 4 == 3) {
        out[b + 12] = 0;
        out[b + 13] = 0;
        out[b + 14] = 0;
    }
    if (tid % 4 >= 2) {
        out[b + 15] = 0;
        out[b + 16] = 0;
    }
}
