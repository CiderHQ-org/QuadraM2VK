// `pow` at every base whose sign or magnitude is an edge, crossed with every exponent whose
// integrality or parity is one.
//
// `air.pow` and `air.fast_pow` are 984 of the 14579 corpus sources and both lower through one
// hand-roll, `lower_metal_pow`: GLSL.std.450 `Pow` leaves a negative base undefined, so the
// translator computes `pow(|x|, y)` and reapplies a sign the exponent's parity earns. Everything
// that hand-roll can get wrong lives in the corners this kernel enumerates, and nothing in the
// corpus can reach them -- a shader that calls `pow` on a texture sample never hands it -0 or -inf.
//
// The 7 x 12 grid is the whole input space of the guard: bases +0, -0, -2, -1, -inf, +inf, +2
// against exponents 3, -3, 2, -2, 0.5, 2.6, 0, -0, +inf, -inf, qNaN, 1. Every combination of
// (sign of base, finiteness of base, integrality of exponent, parity of exponent, sign of exponent)
// appears, so a lowering that tests `x < 0` where it should test the sign bit, or that refuses a
// non-integer exponent without asking whether the base is finite, lands on a lane that separates it.
//
// The base +2 substitutes exponent 4 for 0.5 and -1 for 2.6, the only two lanes whose answer would
// be irrational; every other lane is exactly 0, +-1, +-2, 4, +-8, 0.125, 0.25, 16, +-inf or NaN, so
// the case compares exact bytes and needs no tolerance. Both spellings are asked at float width
// because `precise::pow` and `fast::pow` are different AIR symbols through the same lowering, and
// device measurement says they agree on all 84 lanes; the third channel asks `air.pow.f16`, which
// is what a plain `pow(half, half)` emits (the `precise::`/`fast::` spellings promote to float and
// never reach the half symbol at all).
#include <metal_stdlib>
using namespace metal;

kernel void kernel_pow_signed_zero_and_infinite_base(
    device const float2 *in [[buffer(0)]],
    device float4 *out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    const float b = in[i].x;
    const float e = in[i].y;
    out[i] = float4(precise::pow(b, e), fast::pow(b, e), float(pow(half(b), half(e))), 0.0f);
}
