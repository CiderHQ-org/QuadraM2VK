// Metal's `tanpi(x)` is NOT `tan(float(pi) * x)`, and the gap is much wider than the one that
// already forced `sinpi`/`cospi` off the same pre-multiply. `float(pi) * 0.5` misses pi/2, so the
// pre-multiplied form has no pole at all: it returns a large finite number, and of the WRONG SIGN.
//
// Device-measured on Apple M3 Max / macOS 26.5.2 over 4091 arguments -- a 1/1000 grid on [-2,2]
// plus the quarter integers and the edges -- `tan(pi*x)` disagrees with `tanpi` on 3184 of them.
// `sinpi(x)/cospi(x)` disagrees on 98, none by more than 1e-5 relative, and reproduces every row
// below exactly. `precise::tanpi` and `fast::tanpi` agree on all 4092, so both spellings are here
// and both must answer the same thing.
//
// The rows are the ones the quotient makes STRUCTURALLY exact, so the comparison can be byte-exact
// whatever the underlying sine and cosine do: at a half-integer the residue of the argument
// reduction is 0, so the numerator is an exact +-1 and the denominator an exact +0.0, and IEEE
// division gives the correctly signed infinity; at an integer the numerator is a signed zero and
// the denominator +-1. The quarter-integer rows are the one pair that depends on `sin(pi/4)` and
// `cos(pi/4)` agreeing, which is worth pinning too.
//
// Every argument is read from the buffer so the frontend cannot fold the call away, and the
// non-finite row reports `isnan` rather than a NaN payload.
//
// It discriminates on 18 of its 24 rows. Run against the pre-multiplied lowering the candidate
// answers -2.28e7 for `tanpi(0.5)`, +2.28e7 for `tanpi(-0.5)`, -8.38e7 for `tanpi(1.5)`,
// -96751.9 for `tanpi(100.5)`, +1169.7 for `tanpi(12345.5)`, 8.74e-08 for `tanpi(1)` and -0.0 for
// `tanpi(-0)`. Only the two quarter turns and the `isnan` row survive it.
//
// Device-measured, both spellings:
//   tanpi(0.5) = +inf   tanpi(-0.5) = -inf   tanpi(1.5) = -inf
//   tanpi(100.5) = +inf tanpi(12345.5) = -inf
//   tanpi(+0) = +0      tanpi(-0) = +0       tanpi(1) = +0   tanpi(-1) = +0
//   tanpi(0.25) = 1     tanpi(-0.25) = -1    tanpi(+inf) = NaN
#include <metal_stdlib>
using namespace metal;

kernel void kernel_tanpi_has_poles_at_the_half_integers(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]])
{
    float half_turn = in[0];
    float minus_half = in[1];
    float three_halves = in[2];
    float hundred_and_a_half = in[3];
    float far_half = in[4];
    float zero = in[5];
    float minus_zero = in[6];
    float one = in[7];
    float minus_one = in[8];
    float quarter = in[9];
    float minus_quarter = in[10];
    float positive_infinity = in[11];

    // air.fast_tanpi.f32 -- the poles.
    out[0] = tanpi(half_turn);
    out[1] = tanpi(minus_half);
    out[2] = tanpi(three_halves);
    out[3] = tanpi(hundred_and_a_half);
    out[4] = tanpi(far_half);
    // ... the exact zeros, including both signed zero arguments.
    out[5] = tanpi(zero);
    out[6] = tanpi(minus_zero);
    out[7] = tanpi(one);
    out[8] = tanpi(minus_one);
    // ... the quarter turns and the non-finite guard.
    out[9] = tanpi(quarter);
    out[10] = tanpi(minus_quarter);
    out[11] = isnan(tanpi(positive_infinity)) ? 1.0f : 0.0f;

    // air.tanpi.f32 -- the precise spelling must answer identically.
    out[12] = precise::tanpi(half_turn);
    out[13] = precise::tanpi(minus_half);
    out[14] = precise::tanpi(three_halves);
    out[15] = precise::tanpi(hundred_and_a_half);
    out[16] = precise::tanpi(far_half);
    out[17] = precise::tanpi(zero);
    out[18] = precise::tanpi(minus_zero);
    out[19] = precise::tanpi(one);
    out[20] = precise::tanpi(minus_one);
    out[21] = precise::tanpi(quarter);
    out[22] = precise::tanpi(minus_quarter);
    out[23] = isnan(precise::tanpi(positive_infinity)) ? 1.0f : 0.0f;
}
