// GLSL.std.450 has neither `exp10` nor `log10`, so both are composed. Only ONE of the two
// compositions matters, and it is not the one this fixture is named after.
//
// Device-measured on Apple M3 Max / macOS 26.5.2, one spelling per kernel, at the DEFAULT fast
// math -- which is what this file is compiled with, so `exp10` here is `air.fast_exp10.f32` and
// `log10` is `air.fast_log10.f32`, the only two spellings the corpus contains:
//
//   exp10   powr(10,x), pow(10,x) and exp2(x*log2(10)) are all BIT-IDENTICAL to exp10, on all
//           2077 arguments of a sweep over [-40,40] plus the round integers.
//   log10   log(x)/ln(10) differs on 1995 of a 2011-argument sweep; log2(x)*log10(2) on none.
//
// So the discriminating rows below are the LOG10 rows, not the exp10 rows. `log10(100)` is the
// one a reader will recognize: Metal returns exactly 2, and `log(100)/ln(10)` returns 1.9999999.
// Verified as an A/B -- this case Mismatches against the natural-log lowering and Matches against
// the log2 one.
//
// Metal does NOT return the exact round powers of ten: `exp10(2)` is 99.99999 and `exp10(3)` is
// 999.9998. Those rows pin that, which is worth having even though they do not discriminate.
//
// Every argument is read from the buffer so the frontend cannot fold the call away.
//
// The nine observed rows:
//   exp10(2) = 99.99999, exp10(3) = 999.9998, exp10(0) = 1, exp10(-inf) = 0, exp10(+inf) = +inf
//   log10(100) = 2, log10(0.5) = -0.30103, log10(+0) = -inf, log10(-1) = NaN
#include <metal_stdlib>
using namespace metal;

kernel void kernel_exp10_hits_the_round_powers(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]])
{
    float two = in[0];
    float three = in[1];
    float zero = in[2];
    float negative_infinity = in[3];
    float positive_infinity = in[4];
    float hundred = in[5];
    float one_half = in[6];
    float negative_one = in[7];

    // The round powers of ten: 99.99999 and 999.9998, one ULP below them, as Metal answers.
    out[0] = exp10(two);
    out[1] = exp10(three);
    out[2] = exp10(zero);
    out[3] = exp10(negative_infinity);
    out[4] = exp10(positive_infinity);

    out[5] = log10(hundred);
    out[6] = log10(one_half);
    out[7] = log10(zero);
    out[8] = isnan(log10(negative_one)) ? 1.0f : 0.0f;
}
