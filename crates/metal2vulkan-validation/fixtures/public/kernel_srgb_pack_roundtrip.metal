#include <metal_stdlib>
using namespace metal;

kernel void kernel_srgb_pack_roundtrip(
    device uint *out [[buffer(0)]],
    device const half *hin [[buffer(1)]],
    device const uint *uin [[buffer(2)]])
{
    uint p = pack_half_to_srgb_unorm4x8(half4(hin[0], hin[1], hin[2], hin[3]));
    half4 u = unpack_unorm4x8_srgb_to_half(uin[0]);
    float4 v = unpack_unorm4x8_srgb_to_float(uin[1]);
    out[0] = p;
    out[1] = as_type<ushort>(u.x); out[2] = as_type<ushort>(u.y);
    out[3] = as_type<ushort>(u.z); out[4] = as_type<ushort>(u.w);
    out[5] = as_type<uint>(v.x); out[6] = as_type<uint>(v.y);
    out[7] = as_type<uint>(v.z); out[8] = as_type<uint>(v.w);
}
