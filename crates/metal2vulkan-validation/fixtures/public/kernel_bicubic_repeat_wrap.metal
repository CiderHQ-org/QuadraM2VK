// A bicubic sampler with `address::repeat`, asked for texture coordinates three whole wraps outside
// [0, 1).
//
// This translator has no Vulkan sampler for `filter::bicubic`, so it emulates one: the normalized
// coordinate is multiplied by the extent, clamped to `[-9, size + 9]`, floored, and the resulting
// texel index is put through the address mode by hand. The clamp is justified in
// `clamp_pixel_coord_component_finite` on the grounds that "any value at or below -9 (or at or
// beyond size + 9) still resolves purely through the address mode", which is true of the three
// CLAMP modes and false of the two REPEAT ones: a repeat wraps modulo the extent, and clamping
// first destroys the phase. On this 8-wide texture a pixel coordinate of 24.5 wraps to texel 0, and
// clamped to 17 it wraps to texel 1 with a half-texel blend on top.
//
// The corpus never asks this. All 5499 static-sampler sites decode to zero bicubic samplers with a
// repeat or mirrored-repeat address mode -- the 55 bicubic ones all clamp, and the 676 repeating
// ones are all nearest or linear, which Vulkan expresses natively and which therefore never reach
// the emulation. So this is a case with no corpus reach and a real answer, like the arrayed depth
// gather at `fe1f6dbf`.
//
// **The derivation needs no bicubic arithmetic, which is the point.** `v` is 7/16, so the vertical
// pixel coordinate is 3.5 and its fraction after the half-texel bias is exactly zero; every `u` is
// `(2k+1)/16` plus a whole number of wraps, so the horizontal fraction is exactly zero too. At
// fraction zero the Catmull-Rom weights are exactly (0, 1, 0, 0), so a correct bicubic returns the
// single texel it lands on and nothing depends on the weight polynomials, on their evaluation
// order, or on rounding. Repeat wrapping by whole periods preserves a fraction exactly, so lanes
// 4..7 must return exactly what lanes 0..3 return -- and that pairing holds whatever the tap
// indexing turns out to be, so the case cannot be satisfied by a shifted footprint either.
//
// The texel at (x, y) is (x, y, 4x + y, 1): every one of the 64 is distinct, no two share a
// component pattern, and the third channel separates a column error from a row error.
#include <metal_stdlib>
using namespace metal;

constant sampler bicRepeat(filter::bicubic, address::repeat, coord::normalized);

constant float kU[8] = {
    1.0f / 16.0f,       3.0f / 16.0f,       5.0f / 16.0f,       7.0f / 16.0f,
    1.0f / 16.0f + 3.0f, 3.0f / 16.0f + 3.0f, 5.0f / 16.0f + 3.0f, 7.0f / 16.0f + 3.0f,
};

kernel void kernel_bicubic_repeat_wrap(
    texture2d<float, access::sample> src [[texture(0)]],
    device float4 *out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = src.sample(bicRepeat, float2(kU[tid], 7.0f / 16.0f), level(0));
}
