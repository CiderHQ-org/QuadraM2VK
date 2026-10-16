// Owned fixture (NaN model): `nan_narrow`'s program under the default fast math,
// where the half `+`, `*`, `air.fma`, and `fneg` carry `fast`. A NaN or infinite operand is
// undefined under fast math (MSL 7.1); the case records what Metal answers, the flag state of
// classes B to D that `nan_narrow` (without fast math) does not measure. Lane `i` reads a float
// a, halves ha and hb (the low and high halves of `in[i].y`), and a half hc, and writes four
// words to `out[i]`: half(a), float(half(a)), ha + hb | ha * hb << 16, and fma(ha, hb, hc) |
// -ha << 16. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void nan_narrow_fast(device const uint4 *in [[buffer(0)]],
                            device uint4 *out [[buffer(1)]],
                            uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x);
    half ha = as_type<half>(ushort(in[tid].y)), hb = as_type<half>(ushort(in[tid].y >> 16));
    half hc = as_type<half>(ushort(in[tid].z));
    half n = half(a);
    out[tid] = uint4(uint(as_type<ushort>(n)), as_type<uint>(float(n)),
                     uint(as_type<ushort>(half(ha + hb))) | uint(as_type<ushort>(half(ha * hb))) << 16,
                     uint(as_type<ushort>(fma(ha, hb, hc))) | uint(as_type<ushort>(half(-ha))) << 16);
}
