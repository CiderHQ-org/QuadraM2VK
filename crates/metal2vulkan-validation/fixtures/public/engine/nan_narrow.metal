// Owned fixture (NaN model): which `half` Metal's narrowing of a float NaN answers,
// and which NaN its `half` arithmetic answers. Lane `i` reads a float a, halves ha and hb (the
// low and high halves of `in[i].y`), and a half hc, and writes four words to `out[i]`: half(a)
// (`fptrunc`), float(half(a)), ha + hb | ha * hb << 16, and fma(ha, hb, hc) | -ha << 16 (`fneg`).
// Compiled without fast math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_narrow(device const uint4 *in [[buffer(0)]],
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
