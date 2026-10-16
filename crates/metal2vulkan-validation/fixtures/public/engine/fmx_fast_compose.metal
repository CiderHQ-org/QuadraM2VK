// Owned fixture: how Metal composes its fast-math functions, with the
// default fast math. Each fast function (`air.fast_*`) is written beside the composition MSL
// Table 8.2 gives for it or a candidate one, over the same operands, so the reading shows where
// Apple's result is a composition of other operations and where it is its own approximation:
// `tan` beside `sin * (1 / cos)`, `x / y` beside the reciprocal `1 / y`, `sqrt` beside
// `x * rsqrt(x)`, `pow` beside `exp2(y * log2(x))`, `exp` beside `exp2(x * log2(e))`, `log`
// beside `log2(x) * ln(2)`, and `fmod` beside `trunc(x / y)`.
//
// Thread i reads `in[i]` = float patterns (x, y) and writes 24 words, `out[6i .. 6i + 5]`:
// (tan x, sin x, cos x, 1 / cos x), (x / y, 1 / y, sqrt x, rsqrt x),
// (exp2 x, log2 x, exp x, log x), (pow(x, y), exp2(y * log2 x), powr(x, y), fmod(x, y)),
// (tanh x, atan x, asin x, exp2(x * log2(e))), and
// (trunc(x / y), log2 x * ln(2), x * rsqrt x, sin x * (1 / cos x)).
// The operands cover [-pi, pi], arguments past it up to the largest float, small and denormal
// arguments, positive arguments over the whole range for the logarithms and roots, exponents in
// [-16, 16], divisors from the denormals past 2^126, and the special values. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void fmx_fast_compose(device const uint2 *in [[buffer(0)]],
                             device uint4 *out [[buffer(1)]],
                             uint i [[thread_position_in_grid]])
{
    float2 v = as_type<float2>(in[i]);
    float x = v.x, y = v.y;
    float s = sin(x), co = cos(x);
    float l2 = log2(x);
    out[6 * i + 0] = uint4(as_type<uint>(tan(x)), as_type<uint>(s), as_type<uint>(co),
                           as_type<uint>(1.0f / co));
    out[6 * i + 1] = uint4(as_type<uint>(x / y), as_type<uint>(1.0f / y), as_type<uint>(sqrt(x)),
                           as_type<uint>(rsqrt(x)));
    out[6 * i + 2] = uint4(as_type<uint>(exp2(x)), as_type<uint>(l2), as_type<uint>(exp(x)),
                           as_type<uint>(log(x)));
    out[6 * i + 3] = uint4(as_type<uint>(pow(x, y)), as_type<uint>(exp2(y * l2)),
                           as_type<uint>(powr(x, y)), as_type<uint>(fmod(x, y)));
    out[6 * i + 4] = uint4(as_type<uint>(tanh(x)), as_type<uint>(atan(x)), as_type<uint>(asin(x)),
                           as_type<uint>(exp2(x * M_LOG2E_F)));
    out[6 * i + 5] = uint4(as_type<uint>(trunc(x / y)), as_type<uint>(l2 * M_LN2_F),
                           as_type<uint>(x * rsqrt(x)), as_type<uint>(s * (1.0f / co)));
}
