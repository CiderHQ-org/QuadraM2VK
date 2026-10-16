// Owned fixture: half denormal results of arithmetic and conversions, in a table, without fast math
// (`-fno-fast-math`: no `nsz` on the operations whose result is a zero). The translator rounds
// every `half` result with `OpQuantizeToF16`, which SPIR-V lets flush a result below the smallest
// normal half to either signed zero (`F16DenormFlush`, the comparison class it derives); Metal was
// observed keeping a half denormal from a `bfloat` conversion. This kernel asks the rest: products,
// sums, and differences that land on a half denormal (including a tie to even), conversions from
// `float` into the denormal range (including a tie to zero), widening denormals to `float`, and a
// load/store round trip.
//
// One thread. `a` and `b` are half patterns, `w` float patterns; `out` holds the table's rows
// h[0..11] as zero-extended half patterns in words 0..11 and f[0], f[1] in words 12 and 13
// (the table names two output buffers; a case observes one). Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void half_denormals(
    device const ushort *a [[buffer(0)]],
    device const ushort *b [[buffer(1)]],
    device const uint *w [[buffer(2)]],
    device uint *out [[buffer(3)]])
{
    half a0 = as_type<half>(a[0]), a1 = as_type<half>(a[1]), a2 = as_type<half>(a[2]);
    half a3 = as_type<half>(a[3]), a4 = as_type<half>(a[4]), a5 = as_type<half>(a[5]);
    half a6 = as_type<half>(a[6]);
    half b0 = as_type<half>(b[0]), b1 = as_type<half>(b[1]), b2 = as_type<half>(b[2]);
    half b3 = as_type<half>(b[3]), b4 = as_type<half>(b[4]), b5 = as_type<half>(b[5]);
    half b6 = as_type<half>(b[6]);
    out[0] = uint(as_type<ushort>(a0 * b0));
    out[1] = uint(as_type<ushort>(a1 * b1));
    out[2] = uint(as_type<ushort>(a2 * b2));
    out[3] = uint(as_type<ushort>(a3 * b3));
    out[4] = uint(as_type<ushort>(a4 + b4));
    out[5] = uint(as_type<ushort>(a5 - b5));
    out[6] = uint(as_type<ushort>(a6 * b6));
    out[7] = uint(as_type<ushort>(half(as_type<float>(w[0]))));
    out[8] = uint(as_type<ushort>(half(as_type<float>(w[1]))));
    out[9] = uint(as_type<ushort>(half(as_type<float>(w[2]))));
    out[10] = uint(as_type<ushort>(half(as_type<float>(w[3]))));
    out[11] = uint(as_type<ushort>(as_type<half>(a[9])));
    out[12] = as_type<uint>(float(as_type<half>(a[7])));
    out[13] = as_type<uint>(float(as_type<half>(a[8])));
}
