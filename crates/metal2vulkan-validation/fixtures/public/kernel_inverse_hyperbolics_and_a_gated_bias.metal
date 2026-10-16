// Five `air.*` symbols the corpus reaches once or four times each and that no authored case
// covered: `air.fast_acosh.f32`, `air.fast_asinh.f32`, `air.fast_atanh.f32`, `air.fast_tan.v3f32`
// and `air.normalize_function_constant_predicate.i16`.
//
// The default (fast-math) compile is kept because it is what emits the four `fast_` symbols, so
// every call carries LLVM's fast flags. MSL 8.1 leaves an infinite or NaN operand or result
// undefined under fast math, and under `nsz` either zero is a correct result. MSL Table 8.2
// defines the fast `acosh`, `asinh` and `atanh` by log formulas and the fast `tan` as
// `sin(x) * (1 / cos(x))`, so what they answer is a reading of Apple's approximation inside a
// bound, never a pinned value.
//
// The cases on this fixture ask each function only where its value is zero: acosh(1) in out[0]
// and out[1], asinh(+0) in out[2] to out[4], atanh(+0) in out[5] to out[8], and tan(+0) on the
// three lanes in out[9] to out[11]. Even there the zero is a bounded reading. Table 8.2's formulas
// reduce to log(1), whose fast error is at most 2^-21 absolute on [0.5, 2] (asinh(+0) through the
// fast sqrt(1) = 1 * rsqrt(1), within 2 ulp, and atanh(+0) through the fast division 1 / 1), and
// to sin(+0), at most 2^-13 absolute. A small non-zero result would conform, and so would -0.
// Metal answers +0 in every row.
//
// The first version also asked acosh(+inf), asinh(-0), asinh(-inf), atanh(-0), atanh(+-1),
// tan(-0) and tan(2^-20). The first seven are undefined under these flags (Metal answered
// asinh(-inf) with NaN, its log formula at -inf), and tan(2^-20) is bounded, not pinned
//.
//
// `has_bias` is a `short` function constant that gates the `bias` argument. That is the spelling
// that makes Metal emit `air.normalize_function_constant_predicate.i16` into `air.static_init`:
// the constant is normalized to 0 or 1 before anything reads it. The three cases on this fixture
// set it to 0, 1 and 2, and the 1 and 2 cases produce byte-identical output -- which is the whole
// content of the word "normalize".
#include <metal_stdlib>
using namespace metal;

constant short has_bias [[function_constant(0)]];

kernel void kernel_inverse_hyperbolics_and_a_gated_bias(
    device const float* fin [[buffer(0)]],
    device const float3* vin [[buffer(1)]],
    device const float* bias [[buffer(2), function_constant(has_bias)]],
    device uint* out [[buffer(3)]])
{
    // air.fast_acosh.f32 -- the cases ask acosh(1) in both rows.
    out[0] = as_type<uint>(acosh(fin[0]));
    out[1] = as_type<uint>(acosh(fin[1]));

    // air.fast_asinh.f32 -- asinh(+0) in all three rows.
    out[2] = as_type<uint>(asinh(fin[2]));
    out[3] = as_type<uint>(asinh(fin[3]));
    out[4] = as_type<uint>(asinh(fin[4]));

    // air.fast_atanh.f32 -- atanh(+0) in all four rows.
    out[5] = as_type<uint>(atanh(fin[5]));
    out[6] = as_type<uint>(atanh(fin[6]));
    out[7] = as_type<uint>(atanh(fin[7]));
    out[8] = as_type<uint>(atanh(fin[8]));

    // air.fast_tan.v3f32 -- tan(+0) on all three lanes.
    float3 t = tan(vin[0]);
    out[9] = as_type<uint>(t.x);
    out[10] = as_type<uint>(t.y);
    out[11] = as_type<uint>(t.z);

    // air.normalize_function_constant_predicate.i16 -- the `short` function constant that gates the
    // `bias` argument. Metal normalizes it to 0 or 1 in `air.static_init` before anything reads it.
    float v = fin[9];
    if (has_bias) { v += bias[0]; }
    out[12] = as_type<uint>(v);
}
