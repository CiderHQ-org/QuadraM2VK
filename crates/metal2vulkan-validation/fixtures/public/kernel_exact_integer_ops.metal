#include <metal_stdlib>
using namespace metal;

kernel void kernel_exact_integer_ops(
    device uint *out [[buffer(0)]],
    device const uint *in [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint4 v = uint4(in[0], in[1], in[2], in[3]);
    out[0] = uint(reverse_bits(int(v.x)));
    out[1] = uint(rotate(int(v.x), int(v.y)));
    uint4 pc = popcount(v);
    out[2] = pc.x + (pc.y << 8) + (pc.z << 16) + (pc.w << 24);
    uint2 cl = clamp(uint2(v.x, v.y), uint2(in[4], in[5]), uint2(in[6], in[7]));
    out[3] = cl.x; out[4] = cl.y;
    half a = half(as_type<short>(ushort(in[8])));
    half b = half(as_type<short>(ushort(in[9])));
    half c = half(as_type<short>(ushort(in[10])));
    out[5] = uint(int(median3(a, b, c)));
    out[6] = uint(int(ctz(short(in[11]))));
    out[7] = uint(int(max(short(in[12]), short(in[13]))));
}
