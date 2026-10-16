// Owned fixture: Metal's single-precision math library without fast math
// (`-fno-fast-math`), so every call is the precise `air.<name>.f32` and MSL Table 8.1 bounds it.
// The probe reads, function by function, how far Metal and each device are from the exact value,
// against Apple's stated bound. `trn_math_fast` is the same program with the default fast math.
// metal-fast-math: off
//
// Thread i reads `in[i]` = float patterns (x, y) and writes 32 words, `out[8i .. 8i + 7]`:
// (sin x, cos x, tan x, sincos's sine), (sincos's cosine, asin x, acos x, atan x),
// (atan2(y, x), sinh x, cosh x, tanh x), (asinh x, acosh x, atanh x, exp x),
// (exp2 x, exp10 x, log x, log2 x), (log10 x, pow(x, y), powr(x, y), sqrt x),
// (rsqrt x, fmod(x, y), sinpi x, cospi x), and (tanpi x, x / y, atan2(x, y), y).
// The operands (the case's rationale lists them) cover the specials, [-1, 1] densely, the
// neighbours of multiples of pi/2, every binade, [0.5, 2] and the neighbours of 1, the edges of
// the exponentials, the hyperbolics' ranges, pow's exponents, and large trig arguments. Loop-free.
//
// sincos's cosine, `c`, is declared ahead of the switch's only label (on i >> 31, zero for every
// thread), so every path into the switch bypasses its declaration and Clang emits no lifetime
// marker for it (as in ob2_sized_array.metal): LLVM 22 and later read a marker in a one-argument
// form, so a fixture holding one may decode to other type numbers under each `llvm-as` major a
// gate assembles it with.
#include <metal_stdlib>
using namespace metal;

static uint b(float v) { return as_type<uint>(v); }

kernel void trn_math_precise(device const uint2 *in [[buffer(0)]],
                             device uint4 *out [[buffer(1)]],
                             uint i [[thread_position_in_grid]])
{
    switch (i >> 31) {
        float c;
    default:
        float2 v = as_type<float2>(in[i]);
        float x = v.x, y = v.y;
        float s = sincos(x, c);
        out[8 * i + 0] = uint4(b(sin(x)), b(cos(x)), b(tan(x)), b(s));
        out[8 * i + 1] = uint4(b(c), b(asin(x)), b(acos(x)), b(atan(x)));
        out[8 * i + 2] = uint4(b(atan2(y, x)), b(sinh(x)), b(cosh(x)), b(tanh(x)));
        out[8 * i + 3] = uint4(b(asinh(x)), b(acosh(x)), b(atanh(x)), b(exp(x)));
        out[8 * i + 4] = uint4(b(exp2(x)), b(exp10(x)), b(log(x)), b(log2(x)));
        out[8 * i + 5] = uint4(b(log10(x)), b(pow(x, y)), b(powr(x, y)), b(sqrt(x)));
        out[8 * i + 6] = uint4(b(rsqrt(x)), b(fmod(x, y)), b(sinpi(x)), b(cospi(x)));
        out[8 * i + 7] = uint4(b(tanpi(x)), b(x / y), b(atan2(x, y)), b(y));
    }
}
