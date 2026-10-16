// Owned fixture: how Metal rounds a normalized pack. The translator rounds the
// exact product `x * (2^N - 1)` to nearest even (measured for `rgb10a2` by the old engine),
// rounds the `f32` product for `rgb565` (measured, case the-rgb565-pack-at-the-exact-tie),
// and the 8- and 16-bit formats are unmeasured. The float inputs are the floats whose `f32`
// product is exactly a half-integer while the exact product is not, for every tie of the 8-bit,
// 5-bit, 6-bit, and 2-bit channels and 64 ties of each 10- and 16-bit one: rounding the `f32`
// product to even and rounding the exact product disagree on about half of them, and rounding the
// `f32` product half up disagrees with rounding it to even on the rest. Snorm formats get both
// signs (half up and half away from zero differ below zero). The half inputs are the halves
// nearest to every 8-bit, 5-bit, and 6-bit tie and to 64 10-bit ones (a product formed in half
// precision would round them apart) and every half whose `f32` product with 65535 or 32767
// rounds onto a half-integer. Both get the specials (signed zeros, denormals, 1 +- ulp, out of
// range, infinities, NaNs). Each input sits in a lane whose channel it was chosen for (a 6-bit
// tie in the `565` green lane, a 2-bit tie in alpha). Compiled with the default fast math, as the
// corpus's packs are (pack calls carry no fast-math flags).
//
// Thread i packs `in[i]` and `hin[i]` into every normalized format and writes 18 words: for the
// float vector, `unorm4x8`, `snorm4x8`, `unorm2x16(xy)`, `unorm2x16(zw)`, `snorm2x16(xy)`,
// `snorm2x16(zw)`, `unorm10a2`, `unorm565(xyz)` (zero-extended), and `srgb_unorm4x8` (whose alpha
// is a linear 8-bit channel); then the same nine for the half vector. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a1_pack_rounding(
    device const float4 *in [[buffer(0)]],
    device const half4 *hin [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    float4 v = in[i];
    half4 h = hin[i];
    device uint *o = out + 18 * i;
    o[0] = pack_float_to_unorm4x8(v);
    o[1] = pack_float_to_snorm4x8(v);
    o[2] = pack_float_to_unorm2x16(v.xy);
    o[3] = pack_float_to_unorm2x16(v.zw);
    o[4] = pack_float_to_snorm2x16(v.xy);
    o[5] = pack_float_to_snorm2x16(v.zw);
    o[6] = pack_float_to_unorm10a2(v);
    o[7] = uint(pack_float_to_unorm565(v.xyz));
    o[8] = pack_float_to_srgb_unorm4x8(v);
    o[9] = pack_half_to_unorm4x8(h);
    o[10] = pack_half_to_snorm4x8(h);
    o[11] = pack_half_to_unorm2x16(h.xy);
    o[12] = pack_half_to_unorm2x16(h.zw);
    o[13] = pack_half_to_snorm2x16(h.xy);
    o[14] = pack_half_to_snorm2x16(h.zw);
    o[15] = pack_half_to_unorm10a2(h);
    o[16] = uint(pack_half_to_unorm565(h.xyz));
    o[17] = pack_half_to_srgb_unorm4x8(h);
}
