// Owned fixture: what Metal answers for the precise trigonometric
// functions called in a module compiled with the default fast math, whose calls then carry every
// fast-math flag, at NaN operands of each payload and sign. Thread `i` reads one float pattern x
// from `in[i]` and writes 12 words to `out[3i .. 3i + 2]`: precise::sin(x), precise::cos(x), the
// sine and cosine of precise::sincos(x), precise::tan(x), precise::asin(x), precise::acos(x),
// precise::atan(x), precise::asinh(x), precise::atanh(x), precise::atan2(x, 1), and x. Loop-free.
//
// sincos's cosine, `c`, is declared ahead of the switch's only label (on tid >> 31, zero for
// every thread), so every path into the switch bypasses its declaration and Clang emits no
// lifetime marker for it (as in mlo_half_trig_inf.metal).
#include <metal_stdlib>
using namespace metal;

kernel void dfl_precise_trig_fast(device const uint *in [[buffer(0)]],
                                  device uint4 *out [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]])
{
    switch (tid >> 31) {
        float c;
    default:
        float x = as_type<float>(in[tid]);
        float s = precise::sincos(x, c);
        device uint4 *o = out + 3 * tid;
        o[0] = uint4(as_type<uint>(precise::sin(x)), as_type<uint>(precise::cos(x)),
                     as_type<uint>(s), as_type<uint>(c));
        o[1] = uint4(as_type<uint>(precise::tan(x)), as_type<uint>(precise::asin(x)),
                     as_type<uint>(precise::acos(x)), as_type<uint>(precise::atan(x)));
        o[2] = uint4(as_type<uint>(precise::asinh(x)), as_type<uint>(precise::atanh(x)),
                     as_type<uint>(precise::atan2(x, 1.0f)), in[tid]);
    }
}
