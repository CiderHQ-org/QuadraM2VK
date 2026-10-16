// Owned fixture: `fmx2_divide_fast` without fast math (`-fno-fast-math`),
// where the plain operator carries no flag and only `fast::divide` grants some. Each quotient
// divides by its own load of the divisor (every copy holds the same value), except in words 4
// and 5:
//
//   words   operation                          flags on the `fdiv`
//   0, 1    x / y, 1 / y                       none
//   2, 3    fast::divide(x, y), (1, y)         reassoc nsz arcp contract afn
//   4, 5    fast::divide(x, y), (z, y), one    reassoc nsz arcp contract afn (a shared divisor)
//           load of y
//   6, 7    fast::sincos(t): sine, cosine      (`air.fast_sincos`, no flag)
//   8, 9    fast::sin(t), fast::cos(t)         (`air.fast_sin`, `air.fast_cos`, no flag)
//
// Thread i reads `in[16 i + j]`: (x, y, y) for words 0-1 at j = 0-2 and words 2-3 at j = 3-5
// (the second `y` is the reciprocal's own load), (x, z, y) at j = 6-8, and t at j = 9, 10, 11
// for the sine and cosine, `fast::sin`, and `fast::cos`. It writes 10 words, `out[16 i + w]`;
// words 10-15 keep the output's initial bytes. Loop-free.
//
// sincos's cosine, `c`, is declared ahead of the switch's only label (on i >> 31, zero for every
// thread), so every path into the switch bypasses its declaration and Clang emits no lifetime
// marker for it (as in trn_math_fast.metal): LLVM 22 and later read a marker in a one-argument
// form, so a fixture holding one may decode to other type numbers under each `llvm-as` major a
// gate assembles it with.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void fmx2_divide_precise(device const float *in [[buffer(0)]],
                                device uint *out [[buffer(1)]],
                                uint i [[thread_position_in_grid]])
{
    switch (i >> 31) {
        float c;
    default:
        device const float *v = in + 16 * i;
        device uint *o = out + 16 * i;
        o[0] = as_type<uint>(v[0] / v[1]);
        o[1] = as_type<uint>(1.0f / v[2]);
        o[2] = as_type<uint>(fast::divide(v[3], v[4]));
        o[3] = as_type<uint>(fast::divide(1.0f, v[5]));
        float y = v[8];
        o[4] = as_type<uint>(fast::divide(v[6], y));
        o[5] = as_type<uint>(fast::divide(v[7], y));
        float s = fast::sincos(v[9], c);
        o[6] = as_type<uint>(s);
        o[7] = as_type<uint>(c);
        o[8] = as_type<uint>(fast::sin(v[10]));
        o[9] = as_type<uint>(fast::cos(v[11]));
    }
}
