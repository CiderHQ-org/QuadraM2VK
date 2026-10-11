#include <metal_stdlib>
using namespace metal;

kernel void kernel_narrow_vector_converts(
    device const float4 *fin [[buffer(0)]],
    device const uint2  *uin [[buffer(1)]],
    device const ulong2 *lin [[buffer(2)]],
    device const int2   *iin [[buffer(3)]],
    device const short4 *hin [[buffer(4)]],
    device uint         *out [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    ulong4 a  = ulong4(fin[tid]);
    uchar2 b  = uchar2(uin[tid]);
    uint2  c  = uint2(lin[tid]);
    half2  d  = half2(iin[tid]);
    short3 s3 = hin[tid].xyz;
    uchar3 e  = uchar3(s3);
    short3 f  = short3(e);
    short2 g  = short2(uin[tid]);
    char2  k  = char2(s3.xy);
    short2 m  = short2(k);
    half   hs = half(iin[tid].x);
    half   hu = half(uin[tid].x);

    uint b0 = tid * 13;
    out[b0 +  0] = uint(a.x & 0xffffffffull);
    out[b0 +  1] = uint(a.x >> 32);
    out[b0 +  2] = uint(a.w & 0xffffffffull);
    out[b0 +  3] = uint(a.w >> 32);
    out[b0 +  4] = uint(b.x) | (uint(b.y) << 8);
    out[b0 +  5] = c.x;
    out[b0 +  6] = c.y;
    out[b0 +  7] = uint(as_type<ushort>(d.x)) | (uint(as_type<ushort>(d.y)) << 16);
    out[b0 +  8] = uint(e.x) | (uint(e.y) << 8) | (uint(e.z) << 16);
    out[b0 +  9] = uint(as_type<ushort>(f.x)) | (uint(as_type<ushort>(f.y)) << 16);
    out[b0 + 10] = uint(as_type<ushort>(g.x)) | (uint(as_type<ushort>(g.y)) << 16);
    out[b0 + 11] = uint(as_type<ushort>(m.x)) | (uint(as_type<ushort>(m.y)) << 16);
    out[b0 + 12] = uint(as_type<ushort>(hs)) | (uint(as_type<ushort>(hu)) << 16);
}
