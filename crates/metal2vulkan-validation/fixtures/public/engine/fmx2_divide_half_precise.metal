// Owned fixture: `fmx2_divide_half_fast` without fast math
// (`-fno-fast-math`): the `fdiv half` of `x / y` carries no flag. Whether Metal rounds the
// quotient once to `half`, rounds a `float` quotient to `half`, or multiplies by a reciprocal
// rounded to `half` first is what the lanes separate. Each quotient divides by its own load of
// the divisor (every copy holds the same value):
//
//   word   operation
//   0      x / y        (the quotient's `half` bits, zero-extended)
//   1      1 / y
//
// Thread i reads `in[4 i + j]`: x, y, y at j = 0-2 (j = 3 unused), and writes `out[4 i + w]`;
// words 2 and 3 keep the output's initial bytes. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void fmx2_divide_half_precise(device const half *in [[buffer(0)]],
                                     device uint *out [[buffer(1)]],
                                     uint i [[thread_position_in_grid]])
{
    device const half *v = in + 4 * i;
    device uint *o = out + 4 * i;
    o[0] = uint(as_type<ushort>(half(v[0] / v[1])));
    o[1] = uint(as_type<ushort>(half(half(1.0h) / v[2])));
}
