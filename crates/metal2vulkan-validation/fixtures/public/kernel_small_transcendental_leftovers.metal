// Six `air.*` symbols that the corpus reaches only one or two times each, and that no authored case
// covered. They are grouped into one kernel because none of them is worth a fixture on its own and
// because their arguments do not interact.
//
// `precise::round` is the one with a claim behind it. Metal's `round` breaks a `.5` tie AWAY FROM
// ZERO, which GLSL.std.450 `Round` explicitly leaves to the implementation -- so the lowering picks
// `RoundEven`'s opposite deliberately, and until now nothing pinned it. The four rows here are the
// ties, in both signs: 0.5 -> 1, -0.5 -> -1, 2.5 -> 3, -2.5 -> -3. `RoundEven` would answer
// 0, -0, 2, -2 on all four.
//
// The half rows are mostly chosen where the answer is exact, because a half transcendental is not
// always byte-authorable through MoltenVK -- its runtime compile does not reliably reach the same
// Metal variant the AIR names. `sqrt` is asked at the perfect squares 0, 0.25, 4 and 1; `sinpi` at
// the half-integers, where the reduction residue is 0 and the answer is an exact 0 or +-1. The
// second `cos` lane is the one deliberate exception: `cos(half(1))` is not exact, and it is here to
// find out whether a half cosine agrees, since only `Tanh` and `Atan2` were known to split by
// width. It DOES: Metal and MoltenVK both answer 0x3853, the correctly rounded half of cos(1).
//
// `sinh`/`cosh` are `fast::` at BOTH widths in SPIRV-Cross and the AIR names `air.fast_*`, so those
// two agree by construction and can take ordinary arguments.
//
// Every argument is read from a buffer so nothing folds. Results are bit-cast into one `uint`
// buffer -- floats whole, halves zero-extended -- so a single output region carries both widths
// without a conversion that would round.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_small_transcendental_leftovers(
    device const float* fin [[buffer(0)]],
    device const half* hin [[buffer(1)]],
    device uint* out [[buffer(2)]])
{
    // air.fast_sinh.f32 and air.fast_cosh.f32
    out[0] = as_type<uint>(sinh(fin[0]));
    out[1] = as_type<uint>(sinh(fin[1]));
    out[2] = as_type<uint>(cosh(fin[0]));
    out[3] = as_type<uint>(cosh(fin[1]));

    // air.round.f32 -- the `.5` ties, away from zero in both signs.
    out[4] = as_type<uint>(precise::round(fin[2]));
    out[5] = as_type<uint>(precise::round(fin[3]));
    out[6] = as_type<uint>(precise::round(fin[4]));
    out[7] = as_type<uint>(precise::round(fin[5]));

    // air.sqrt.v2f16 -- the perfect squares.
    half2 roots_low = sqrt(half2(hin[0], hin[1]));
    half2 roots_high = sqrt(half2(hin[2], hin[3]));
    out[8] = uint(as_type<ushort>(roots_low.x));
    out[9] = uint(as_type<ushort>(roots_low.y));
    out[10] = uint(as_type<ushort>(roots_high.x));
    out[11] = uint(as_type<ushort>(roots_high.y));

    // air.cos.v2f16 -- one exact lane at zero and one that is not exact, on purpose.
    half2 cosines = cos(half2(hin[4], hin[5]));
    out[12] = uint(as_type<ushort>(cosines.x));
    out[13] = uint(as_type<ushort>(cosines.y));

    // air.sinpi.f16 -- the half-integers, where the residue is 0 and the answer is exact.
    out[14] = uint(as_type<ushort>(sinpi(hin[6])));
    out[15] = uint(as_type<ushort>(sinpi(hin[7])));
    out[16] = uint(as_type<ushort>(sinpi(hin[8])));
    out[17] = uint(as_type<ushort>(sinpi(hin[9])));
}
