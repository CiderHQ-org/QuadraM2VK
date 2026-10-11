// Metal specifies its plain `fmin`/`fmax` NaN-aware: "if one argument is a NaN, returns the other
// argument; if both arguments are NaNs, returns a NaN". `clamp` is `fmin(fmax(x, lo), hi)` and
// `saturate` is `clamp(x, 0, 1)`, so both inherit it -- a NaN clamps to the FLOOR, not to zero and
// not to a NaN.
//
// That sentence is GLSL.std.450 `NMin`/`NMax`/`NClamp` word for word, and it is what the emitter
// now reaches for a plain (non-`fast_`) symbol. `FMin`/`FMax`/`FClamp` compute the same thing for
// ordered operands but leave the NaN case UNDEFINED, so nothing but a case pins which one we owe.
//
// Every NaN is read from the input buffer: a literal one would be folded by the frontend and the
// kernel would stop testing the hardware. The floor is 2.5 rather than 0 so that "clamps to the
// floor" is distinguishable from "answers zero", and the ceiling 7.25 so an operand-swapped clamp
// answers differently.
//
// `out[7]` asks only whether the both-NaN answer IS a NaN, never which one: measured on an Apple
// M3 Max, `fmax(qNaN, -qNaN)` is `0xffc00000` -- the SECOND operand, sign and payload intact -- and
// an implementation returning the first is equally correct, so comparing the bits would be
// asserting a coincidence.
//
// Device-measured on Apple M3 Max / macOS 26.5.2, compiled with `-fno-fast-math` so the symbols are
// the plain ones: 3.0 for all four min/max rows, 2.5 for both NaN clamps and for the below-range
// clamp, 0.0 for the saturate, 1.0 for the is-a-NaN probe, and 7.25 for the above-range clamp.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_a_nan_loses_to_a_number(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]])
{
    float nan_positive = in[0];
    float nan_negative = in[1];
    float three = in[2];
    float floor_edge = in[3];
    float ceiling_edge = in[4];
    float above_range = in[5];
    float below_range = in[6];

    // The number wins whichever side the NaN arrives on.
    out[0] = fmax(nan_positive, three);
    out[1] = fmax(three, nan_positive);
    out[2] = fmin(nan_positive, three);
    out[3] = fmin(three, nan_positive);

    // `fmax(NaN, lo)` is `lo` and `fmin(lo, hi)` is `lo`, so a NaN clamps to the floor.
    out[4] = clamp(nan_positive, floor_edge, ceiling_edge);
    out[5] = clamp(nan_negative, floor_edge, ceiling_edge);

    // `saturate` is that same clamp against 0 and 1, so a NaN saturates to zero.
    out[6] = saturate(nan_positive);

    // Two NaNs in, a NaN out -- which one is not specified and is not asserted.
    out[7] = isnan(fmax(nan_positive, nan_negative)) ? 1.0f : 0.0f;

    // Ordinary clamping still has to work, or the NaN rows above prove nothing.
    out[8] = clamp(above_range, floor_edge, ceiling_edge);
    out[9] = clamp(below_range, floor_edge, ceiling_edge);
}
