#include <metal_stdlib>
using namespace metal;

kernel void kernel_convert_fma_fract(
    device uint *out [[buffer(0)]],
    device const float *fin [[buffer(1)]],
    device const ushort *sin_ [[buffer(2)]],
    device const half *hin [[buffer(3)]])
{
    uchar4 a = uchar4(float4(fin[0], fin[1], fin[2], fin[3]));
    int3 b = int3(ushort3(sin_[0], sin_[1], sin_[2]));
    bool2 flags = bool2(sin_[3] != 0, sin_[4] != 0);
    half2 c = half2(flags);
    half3 d = fma(half3(hin[0], hin[1], hin[2]),
                  half3(hin[3], hin[4], hin[5]),
                  half3(hin[6], hin[7], hin[8]));
    half3 e = fract(half3(hin[9], hin[10], hin[11]));

    out[0] = uint(a.x); out[1] = uint(a.y); out[2] = uint(a.z); out[3] = uint(a.w);
    out[4] = as_type<uint>(b.x); out[5] = as_type<uint>(b.y); out[6] = as_type<uint>(b.z);
    out[7] = uint(as_type<ushort>(c.x)); out[8] = uint(as_type<ushort>(c.y));
    out[9]  = uint(as_type<ushort>(d.x));
    out[10] = uint(as_type<ushort>(d.y));
    out[11] = uint(as_type<ushort>(d.z));
    out[12] = uint(as_type<ushort>(e.x));
    out[13] = uint(as_type<ushort>(e.y));
    out[14] = uint(as_type<ushort>(e.z));
}
