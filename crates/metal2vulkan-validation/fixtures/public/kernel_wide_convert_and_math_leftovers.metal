// Sixteen `air.*` symbols the corpus reaches a handful of times each and that no authored case
// covered: the wide-vector conversions convert.f.{v16f32.f.v16f16, v16bf16.f.v16f16, v8bf16.f.v8f16,
// v8f16.f.v8bf16, v4bf16.f.v4f16, v4f16.s.v4i32}, and powr.v4f16, max.u.v3i32, max.s.v3i32,
// abs.s.v3i32, fract.v2f16, fmin.v3f32, fmax.v3f32, floor.v3f16, fast_fmedian3.f32 and
// fast_cospi.f32. One thread, no loop, every argument read from a buffer so nothing folds.
//
// The half-to-bfloat rows are the ones with a claim behind them. Half to binary32 is exact, so
// half to bfloat rounds exactly ONCE, and the sixteen inputs walk every case of that rounding:
// exact, round down, round up, a tie whose kept bit is even (rounds down) and a tie whose kept bit
// is odd (rounds up), the largest half (which carries out to 65536), both zeros, both infinities,
// the smallest normal and two subnormals -- a half subnormal is an ordinary bfloat, because bfloat
// carries binary32's exponent range. The reverse direction has the opposite hazard: a bfloat's
// 8-bit significand always fits in a half, but its exponent range does not, so `1e30` becomes
// infinity and `1e-30` becomes zero.
//
// `air.convert.f.v4f16.s.v4i32` asks two ties (2049 -> 2048 and -2051 -> -2052, both to even) and
// one overflow (70000 -> infinity). `max.u.v3i32` has 0xFFFFFFFF in lane 1 and `max.s.v3i32` has a
// negative in lane 0, so each would answer the other's lane wrong if the comparison lost its
// signedness. `fract` lane 0 is the negated smallest normal half, where `x - floor(x)` rounds to 1
// and Metal must clamp to the largest half below it. `fmin`/`fmax` lane 0 has a NaN operand, which
// both must ignore and return the other side.
//
// The rows that could not be asked exactly are asked where the answer is: `powr` at power-of-two
// bases with dyadic exponents, so log2 and exp2 are both exact; `cospi` at the integers and one
// half-integer; `median3` is a pure selection, so its fast variant cannot differ at all.
//
// Results are bit-cast into one `uint` buffer -- floats whole, halves and bfloats zero-extended --
// so a single output region carries every width without a conversion that would round.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_wide_convert_and_math_leftovers(
    device const half* hs [[buffer(0)]],
    device const bfloat* bs [[buffer(1)]],
    device const int* is [[buffer(2)]],
    device const uint* us [[buffer(3)]],
    device const float* fs [[buffer(4)]],
    device uint* out [[buffer(5)]])
{
    vec<half, 16> h16 = vec<half, 16>(hs[0], hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7], hs[8], hs[9], hs[10], hs[11], hs[12], hs[13], hs[14], hs[15]);
    vec<half, 8> h8 = vec<half, 8>(hs[16], hs[17], hs[18], hs[19], hs[20], hs[21], hs[22], hs[23]);
    vec<bfloat, 8> b8 = vec<bfloat, 8>(bs[0], bs[1], bs[2], bs[3], bs[4], bs[5], bs[6], bs[7]);
    half4 h4 = half4(hs[24], hs[25], hs[26], hs[27]);

    // air.convert.f.v16f32.f.v16f16 -- widening, exact at every half.
    vec<float, 16> wf = vec<float, 16>(h16);
    out[0] = as_type<uint>(wf[0]);
    out[1] = as_type<uint>(wf[1]);
    out[2] = as_type<uint>(wf[2]);
    out[3] = as_type<uint>(wf[3]);
    out[4] = as_type<uint>(wf[4]);
    out[5] = as_type<uint>(wf[5]);
    out[6] = as_type<uint>(wf[6]);
    out[7] = as_type<uint>(wf[7]);
    out[8] = as_type<uint>(wf[8]);
    out[9] = as_type<uint>(wf[9]);
    out[10] = as_type<uint>(wf[10]);
    out[11] = as_type<uint>(wf[11]);
    out[12] = as_type<uint>(wf[12]);
    out[13] = as_type<uint>(wf[13]);
    out[14] = as_type<uint>(wf[14]);
    out[15] = as_type<uint>(wf[15]);

    // air.convert.f.v16bf16.f.v16f16 -- narrows an 11-bit significand to 8, rounding once.
    vec<bfloat, 16> wb = vec<bfloat, 16>(h16);
    out[16] = uint(as_type<ushort>(wb[0]));
    out[17] = uint(as_type<ushort>(wb[1]));
    out[18] = uint(as_type<ushort>(wb[2]));
    out[19] = uint(as_type<ushort>(wb[3]));
    out[20] = uint(as_type<ushort>(wb[4]));
    out[21] = uint(as_type<ushort>(wb[5]));
    out[22] = uint(as_type<ushort>(wb[6]));
    out[23] = uint(as_type<ushort>(wb[7]));
    out[24] = uint(as_type<ushort>(wb[8]));
    out[25] = uint(as_type<ushort>(wb[9]));
    out[26] = uint(as_type<ushort>(wb[10]));
    out[27] = uint(as_type<ushort>(wb[11]));
    out[28] = uint(as_type<ushort>(wb[12]));
    out[29] = uint(as_type<ushort>(wb[13]));
    out[30] = uint(as_type<ushort>(wb[14]));
    out[31] = uint(as_type<ushort>(wb[15]));

    // air.convert.f.v8bf16.f.v8f16
    vec<bfloat, 8> nb = vec<bfloat, 8>(h8);
    out[32] = uint(as_type<ushort>(nb[0]));
    out[33] = uint(as_type<ushort>(nb[1]));
    out[34] = uint(as_type<ushort>(nb[2]));
    out[35] = uint(as_type<ushort>(nb[3]));
    out[36] = uint(as_type<ushort>(nb[4]));
    out[37] = uint(as_type<ushort>(nb[5]));
    out[38] = uint(as_type<ushort>(nb[6]));
    out[39] = uint(as_type<ushort>(nb[7]));

    // air.convert.f.v8f16.f.v8bf16 -- wider significand, much narrower exponent range.
    vec<half, 8> nh = vec<half, 8>(b8);
    out[40] = uint(as_type<ushort>(nh[0]));
    out[41] = uint(as_type<ushort>(nh[1]));
    out[42] = uint(as_type<ushort>(nh[2]));
    out[43] = uint(as_type<ushort>(nh[3]));
    out[44] = uint(as_type<ushort>(nh[4]));
    out[45] = uint(as_type<ushort>(nh[5]));
    out[46] = uint(as_type<ushort>(nh[6]));
    out[47] = uint(as_type<ushort>(nh[7]));

    // air.convert.f.v4bf16.f.v4f16
    bfloat4 qb = bfloat4(h4);
    out[48] = uint(as_type<ushort>(qb[0]));
    out[49] = uint(as_type<ushort>(qb[1]));
    out[50] = uint(as_type<ushort>(qb[2]));
    out[51] = uint(as_type<ushort>(qb[3]));

    // air.convert.f.v4f16.s.v4i32
    half4 qh = half4(int4(is[0], is[1], is[2], is[3]));
    out[52] = uint(as_type<ushort>(qh[0]));
    out[53] = uint(as_type<ushort>(qh[1]));
    out[54] = uint(as_type<ushort>(qh[2]));
    out[55] = uint(as_type<ushort>(qh[3]));

    // air.powr.v4f16 -- power-of-two bases and dyadic exponents, so log2 and exp2 are exact.
    half4 pb = half4(hs[28], hs[29], hs[30], hs[31]);
    half4 pe = half4(hs[32], hs[33], hs[34], hs[35]);
    half4 pr = powr(pb, pe);
    out[56] = uint(as_type<ushort>(pr[0]));
    out[57] = uint(as_type<ushort>(pr[1]));
    out[58] = uint(as_type<ushort>(pr[2]));
    out[59] = uint(as_type<ushort>(pr[3]));

    // air.max.u.v3i32 -- lane 1 is 0xFFFFFFFF, which a signed compare would answer wrong.
    uint3 mu = max(uint3(us[0], us[1], us[2]), uint3(us[3], us[4], us[5]));
    out[60] = mu.x; out[61] = mu.y; out[62] = mu.z;

    // air.max.s.v3i32 -- lane 0 is negative, which an unsigned compare would answer wrong.
    int3 ms = max(int3(is[4], is[5], is[6]), int3(is[7], is[8], is[9]));
    out[63] = as_type<uint>(ms.x); out[64] = as_type<uint>(ms.y); out[65] = as_type<uint>(ms.z);

    // air.abs.s.v3i32
    int3 ab = abs(int3(is[10], is[11], is[12]));
    out[66] = as_type<uint>(ab.x); out[67] = as_type<uint>(ab.y); out[68] = as_type<uint>(ab.z);

    // air.fract.v2f16 -- lane 0 is the smallest normal half negated, where x - floor(x) rounds
    // to 1 and Metal must clamp to the largest half below 1.
    half2 fr = fract(half2(hs[36], hs[37]));
    out[69] = uint(as_type<ushort>(fr.x));
    out[70] = uint(as_type<ushort>(fr.y));

    // air.fmin.v3f32 / air.fmax.v3f32 -- lane 0 has a NaN operand, which both must ignore.
    float3 lo = precise::fmin(float3(fs[0], fs[1], fs[2]), float3(fs[3], fs[4], fs[5]));
    out[71] = as_type<uint>(lo.x); out[72] = as_type<uint>(lo.y); out[73] = as_type<uint>(lo.z);
    float3 hi = precise::fmax(float3(fs[0], fs[1], fs[2]), float3(fs[3], fs[4], fs[5]));
    out[74] = as_type<uint>(hi.x); out[75] = as_type<uint>(hi.y); out[76] = as_type<uint>(hi.z);

    // air.floor.v3f16
    half3 fl = floor(half3(hs[38], hs[39], hs[40]));
    out[77] = uint(as_type<ushort>(fl.x));
    out[78] = uint(as_type<ushort>(fl.y));
    out[79] = uint(as_type<ushort>(fl.z));

    // air.fast_fmedian3.f32 -- a pure selection, so the fast variant cannot differ.
    out[80] = as_type<uint>(median3(fs[6], fs[7], fs[8]));
    out[81] = as_type<uint>(median3(fs[9], fs[10], fs[11]));

    // air.fast_cospi.f32 at the integers and one half-integer.
    out[82] = as_type<uint>(cospi(fs[12]));
    out[83] = as_type<uint>(cospi(fs[13]));
    out[84] = as_type<uint>(cospi(fs[14]));
    out[85] = as_type<uint>(cospi(fs[15]));
}
