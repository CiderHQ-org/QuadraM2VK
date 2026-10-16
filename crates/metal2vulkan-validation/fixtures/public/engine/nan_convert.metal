// Owned fixture (NaN model): Metal's vector conversions between `float` and `half`
// beside the scalar casts. Lane `i` reads a float4 a (the bits in `in[2i]`) and a half4 h (the
// four halves of `in[2i + 1].xy`, the first in the low half) and writes eight words to
// `out[2i .. 2i + 1]`: half4(a) (two words, x | y << 16 and z | w << 16), half(a.x) |
// half(a.y) << 16 (scalar narrowings), float(h.x) (a scalar widening), then float4(h).
// `nan_convert_fast` is the same program under the default fast math. Compiled without fast
// math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

static uint pack(half lo, half hi)
{
    return uint(as_type<ushort>(lo)) | uint(as_type<ushort>(hi)) << 16;
}

kernel void nan_convert(device const uint4 *in [[buffer(0)]],
                        device uint4 *out [[buffer(1)]],
                        uint tid [[thread_position_in_grid]])
{
    float4 a = as_type<float4>(in[2 * tid]);
    uint2 hw = in[2 * tid + 1].xy;
    half4 h = half4(as_type<half2>(hw.x), as_type<half2>(hw.y));
    half4 n = half4(a);
    out[2 * tid] = uint4(pack(n.x, n.y), pack(n.z, n.w), pack(half(a.x), half(a.y)),
                         as_type<uint>(float(h.x)));
    out[2 * tid + 1] = as_type<uint4>(float4(h));
}
