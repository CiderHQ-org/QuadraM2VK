// Owned fixture: how Metal divides under each fast-math flag set, and how
// `fast::sincos` answers past the last quadrant, compiled with the default fast math. A Metal
// probe found a fast `x / y` equal to `x` times Metal's fast `1 / y`, but its kernel
// also divided by the same `y` elsewhere, which a compiler may turn into one reciprocal shared by
// every quotient. Here each quotient divides by its own load of the divisor (every copy holds
// the same value), so no division shares its divisor, except in words 10 and 11:
//
//   words   operation                          flags on the `fdiv`
//   0, 1    x / y, 1 / y                       fast
//   2, 3    x / y, 1 / y under                 nnan ninf nsz arcp contract afn
//           `reassociate(off)`
//   4, 5    x / y, 1 / y under `contract(off)` reassoc nnan ninf nsz arcp afn
//   6, 7    fast::divide(x, y), (1, y)         reassoc nsz arcp contract afn
//   8, 9    precise::divide(x, y), (1, y)      none
//   10, 11  x / y and z / y, one load of y     fast (a shared divisor)
//   12, 13  fast::sincos(t): sine, cosine      (`air.fast_sincos`, fast)
//   14, 15  fast::sin(t), fast::cos(t)         (`air.fast_sin`, `air.fast_cos`, fast)
//
// Thread i reads `in[24 i + j]`: (x, y, y) for words 0-1 at j = 0-2, words 2-3 at j = 3-5, words
// 4-5 at j = 6-8, words 6-7 at j = 9-11, words 8-9 at j = 12-14 (the second `y` is the
// reciprocal's own load), (x, z, y) at j = 15-17, and t at j = 18, 19, 20 for the sine and
// cosine, `fast::sin`, and `fast::cos`. It writes 16 words, `out[16 i + w]`. Loop-free.
//
// sincos's cosine, `c`, is declared ahead of the switch's only label (on i >> 31, zero for every
// thread), so every path into the switch bypasses its declaration and Clang emits no lifetime
// marker for it (as in trn_math_fast.metal): LLVM 22 and later read a marker in a one-argument
// form, so a fixture holding one may decode to other type numbers under each `llvm-as` major a
// gate assembles it with.
#include <metal_stdlib>
using namespace metal;

kernel void fmx2_divide_fast(device const float *in [[buffer(0)]],
                             device uint *out [[buffer(1)]],
                             uint i [[thread_position_in_grid]])
{
    switch (i >> 31) {
        float c;
    default:
        device const float *v = in + 24 * i;
        device uint *o = out + 16 * i;
        o[0] = as_type<uint>(v[0] / v[1]);
        o[1] = as_type<uint>(1.0f / v[2]);
        {
#pragma clang fp reassociate(off)
            o[2] = as_type<uint>(v[3] / v[4]);
            o[3] = as_type<uint>(1.0f / v[5]);
        }
        {
#pragma clang fp contract(off)
            o[4] = as_type<uint>(v[6] / v[7]);
            o[5] = as_type<uint>(1.0f / v[8]);
        }
        o[6] = as_type<uint>(fast::divide(v[9], v[10]));
        o[7] = as_type<uint>(fast::divide(1.0f, v[11]));
        o[8] = as_type<uint>(precise::divide(v[12], v[13]));
        o[9] = as_type<uint>(precise::divide(1.0f, v[14]));
        float y = v[17];
        o[10] = as_type<uint>(v[15] / y);
        o[11] = as_type<uint>(v[16] / y);
        float s = fast::sincos(v[18], c);
        o[12] = as_type<uint>(s);
        o[13] = as_type<uint>(c);
        o[14] = as_type<uint>(fast::sin(v[19]));
        o[15] = as_type<uint>(fast::cos(v[20]));
    }
}
