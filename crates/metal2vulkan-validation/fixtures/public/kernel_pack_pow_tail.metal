#include <metal_stdlib>
using namespace metal;

kernel void kernel_pack_pow_tail(
    device uint *out [[buffer(0)]],
    device const half *hin [[buffer(1)]],
    device const float *fin [[buffer(2)]],
    device const uint *uin [[buffer(3)]])
{
    half4 p = pow(half4(hin[0], hin[1], hin[2], hin[3]),
                  half4(hin[4], hin[5], hin[6], hin[7]));
    half2 r = powr(half2(hin[8], hin[9]), half2(hin[10], hin[11]));
    float2 fp = fast::pow(float2(fin[0], fin[1]), float2(fin[2], fin[3]));

    uint a = pack_half_to_unorm2x16(half2(hin[12], hin[13]));
    ushort b = pack_half_to_unorm565(half3(hin[14], hin[15], hin[16]));

    float2 g = unpack_unorm2x16_to_float(uin[0]);
    half2 h = unpack_unorm2x16_to_half(uin[1]);
    half4 i = unpack_unorm10a2_to_half(uin[2]);

    out[0] = as_type<ushort>(p.x); out[1] = as_type<ushort>(p.y);
    out[2] = as_type<ushort>(p.z); out[3] = as_type<ushort>(p.w);
    out[4] = as_type<ushort>(r.x); out[5] = as_type<ushort>(r.y);
    out[6] = as_type<uint>(fp.x);  out[7] = as_type<uint>(fp.y);
    out[8] = a; out[9] = uint(b);
    out[10] = as_type<uint>(g.x); out[11] = as_type<uint>(g.y);
    out[12] = as_type<ushort>(h.x); out[13] = as_type<ushort>(h.y);
    out[14] = as_type<ushort>(i.x); out[15] = as_type<ushort>(i.y);
    out[16] = as_type<ushort>(i.z); out[17] = as_type<ushort>(i.w);
}
