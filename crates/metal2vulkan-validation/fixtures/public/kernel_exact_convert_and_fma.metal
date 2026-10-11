#include <metal_stdlib>
using namespace metal;

kernel void kernel_exact_convert_and_fma(
    device uint *out [[buffer(0)]],
    device const float *fin [[buffer(1)]],
    device const uint *uin [[buffer(2)]],
    device const ulong *lin [[buffer(3)]],
    device const half *hin [[buffer(4)]],
    device const ushort *sin_ [[buffer(5)]],
    device const int *iin [[buffer(6)]])
{
    float3 a = float3(fin[0], fin[1], fin[2]);
    float3 b = float3(fin[3], fin[4], fin[5]);
    float3 c = float3(fin[6], fin[7], fin[8]);
    float3 m = fma(a, b, c);

    ulong4 ul = ulong4(lin[0], lin[1], lin[2], lin[3]);
    long4 sl = long4(as_type<long>(lin[4]), as_type<long>(lin[5]),
                     as_type<long>(lin[6]), as_type<long>(lin[7]));
    float4 fu = float4(ul);
    float4 fs = float4(sl);
    float4 tl = float4(fin[9], fin[10], fin[11], fin[12]);
    long4 back = long4(tl);

    int2 i2 = int2(half2(hin[0], hin[1]));
    half3 h3 = half3(ushort3(sin_[0], sin_[1], sin_[2]));
    ushort3 s3 = ushort3(uint3(uin[0], uin[1], uin[2]));
    short4 s4 = short4(float4(fin[13], fin[14], fin[15], fin[16]));

    half2 sat0 = saturate(half2(hin[2], hin[3]));
    half2 sat1 = saturate(half2(hin[4], hin[5]));
    float2 tr = trunc(float2(fin[17], fin[18]));
    float ld = ldexp(fin[19], iin[0]);

    out[0] = as_type<uint>(m.x);
    out[1] = as_type<uint>(m.y);
    out[2] = as_type<uint>(m.z);
    out[3] = as_type<uint>(fu.x); out[4] = as_type<uint>(fu.y);
    out[5] = as_type<uint>(fu.z); out[6] = as_type<uint>(fu.w);
    out[7] = as_type<uint>(fs.x); out[8] = as_type<uint>(fs.y);
    out[9] = as_type<uint>(fs.z); out[10] = as_type<uint>(fs.w);
    ulong4 bw = as_type<ulong4>(back);
    out[11] = uint(bw.x & 0xFFFFFFFFul); out[12] = uint(bw.x >> 32);
    out[13] = uint(bw.y & 0xFFFFFFFFul); out[14] = uint(bw.y >> 32);
    out[15] = uint(bw.z & 0xFFFFFFFFul); out[16] = uint(bw.z >> 32);
    out[17] = uint(bw.w & 0xFFFFFFFFul); out[18] = uint(bw.w >> 32);
    out[19] = as_type<uint>(i2.x);
    out[20] = as_type<uint>(i2.y);
    out[21] = as_type<ushort>(h3.x);
    out[22] = as_type<ushort>(h3.y);
    out[23] = as_type<ushort>(h3.z);
    out[24] = s3.x; out[25] = s3.y; out[26] = s3.z;
    out[27] = uint(as_type<ushort>(s4.x));
    out[28] = uint(as_type<ushort>(s4.y));
    out[29] = uint(as_type<ushort>(s4.z));
    out[30] = uint(as_type<ushort>(s4.w));
    out[31] = as_type<ushort>(sat0.x); out[32] = as_type<ushort>(sat0.y);
    out[33] = as_type<ushort>(sat1.x); out[34] = as_type<ushort>(sat1.y);
    out[35] = as_type<uint>(tr.x); out[36] = as_type<uint>(tr.y);
    out[37] = as_type<uint>(ld);
}
