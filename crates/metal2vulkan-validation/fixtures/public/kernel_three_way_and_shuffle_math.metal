// Fifteen uncovered arithmetic families in one kernel: air.fast_fmin3.{v3f32,v2f32} (2 corpus
// sources each), air.fast_fmax3.{v3f32,v2f32} (2 each), air.fmax3.v3f16 (2), air.fmedian3.v4f16
// (2), air.max.s.v2i16 (2), air.min3.u.i32 (2), air.pow.v4f32 (2), air.quad_shuffle.{f32,v4f32}
// (2 each), air.trunc.f16 (1), air.fast_tanh.{v3f32,v2f32} (2 each) and air.fast_log2.v2f32 (2) --
// 29 sources.
//
// Everything here has an answer that is exact and hand-derivable, which is the point: a three-way
// minimum, a median, a signed maximum and a lane shuffle are all SELECTIONS, so the only way to be
// wrong about them is to select the wrong thing, and that is only visible if the candidates differ.
//
//   * The three float vectors that feed each three-way call cross each other: their componentwise
//     order changes with the thread, so no lane's answer comes from the same argument twice, and a
//     lowering that always took the first, the last, or a two-argument min cannot follow.
//   * `air.max.s.v2i16` is asked where the two derivations disagree. Lane x is `t - 3` against
//     `5 - t`, so it is negative for the first three threads; read as unsigned, `t - 3` is 65533
//     and the maximum flips to it. That is the signedness bug shape 6d8766b3 was, in the family
//     that spells the signedness in the symbol.
//   * Every half is a multiple of 1/2 below 32 and every float an integer or a power of two, so all
//     of them are exact in their own format and the comparison needs no tolerance. `trunc` is asked
//     at negative halves ending in .5 and .25, where truncating toward zero and flooring differ.
//   * `tanh` is asked ONLY at zero, because on this host `fast::tanh` supports no other anchor a
//     derivation can stand on. It does not saturate monotonically -- it returns 1 at 35 and at 37
//     and 0.99999994 at 36 -- and it is not odd: `tanh(x) + tanh(-x)` is between 8e-10 and 8e-8,
//     never zero, at all sixteen arguments tried. `tanh(0) = 0` survives both, and a lowering that
//     answered `cosh`, `exp`, `cos` or `1 - x` still cannot produce zero. The argument is
//     `u - u` rather than a literal so the call cannot fold away.
//     `log2` is asked at powers of two the thread index builds, and `pow` only where
//     `exp2(y * log2(x))` is exact at every step: log2 of 2, 4, 1/2 and 16 are 1, 2, -1 and 4, and
//     each product with its exponent is a whole number.
//   * The dispatch is two quads of one simdgroup, so `quad_shuffle` has a derivable answer rather
//     than a hardware-shaped one, and lanes 0 and 2 are asked so the two quads answer differently.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_three_way_and_shuffle_math(
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    float u = float(tid);

    float3 a1 = float3(u, 8.0f - u, 2.0f * u - 3.0f);
    float3 a2 = float3(u + 16.0f, 3.0f - u, 4.0f * u);
    float3 a3 = float3(2.0f * u - 5.0f, u + 1.0f, 9.0f - u);
    float3 fmin_v3 = fmin3(a1, a2, a3);
    float3 fmax_v3 = fmax3(a1, a2, a3);

    float2 d1 = float2(3.0f * u - 1.0f, 7.0f - 2.0f * u);
    float2 d2 = float2(11.0f - u, 4.0f * u - 6.0f);
    float2 d3 = float2(u + 2.0f, 5.0f - u);
    float2 fmin_v2 = fmin3(d1, d2, d3);
    float2 fmax_v2 = fmax3(d1, d2, d3);

    half h = half(u);
    half3 p1 = half3(h + 0.5h, 6.5h - h, h * 1.5h);
    half3 p2 = half3(h * 2.0h, 1.5h, 9.5h - h);
    half3 p3 = half3(3.5h, h + 4.5h, h - 2.5h);
    half3 hmax_v3 = fmax3(p1, p2, p3);
    half4 hmed_v4 = fmedian3(half4(p1, h - 3.5h), half4(p2, 2.5h), half4(p3, 7.5h - h));

    short2 s1 = short2(short(tid) - 3, short(tid) * 2 - 7);
    short2 s2 = short2(5 - short(tid), short(tid));
    short2 smax = max(s1, s2);

    uint m1 = min3(tid, 5u, 3u);
    uint m2 = min3(tid + 2u, 4u, 9u);

    half t1 = trunc(h - 3.5h);
    half t2 = trunc(-(h * 1.5h) - 0.25h);

    float z = u - u;
    float3 tanh_v3 = tanh(float3(z, z, z));
    float2 tanh_v2 = tanh(float2(z, z));
    float2 log_v2 = log2(float2(float(1u << tid), float(1u << (tid + 2u))));
    float4 pw = precise::pow(float4(2.0f, 4.0f, 0.5f, 16.0f), float4(u, 1.0f, 2.0f, 0.25f));

    float q = u + 100.0f;
    float qs2 = quad_shuffle(q, 2);
    float qs0 = quad_shuffle(q, 0);
    float4 qv = quad_shuffle(float4(u, 2.0f * u, 3.0f * u, 4.0f * u), 1);

    uint b = tid * 38u;
    out[b +  0] = as_type<uint>(fmin_v3.x);
    out[b +  1] = as_type<uint>(fmin_v3.y);
    out[b +  2] = as_type<uint>(fmin_v3.z);
    out[b +  3] = as_type<uint>(fmax_v3.x);
    out[b +  4] = as_type<uint>(fmax_v3.y);
    out[b +  5] = as_type<uint>(fmax_v3.z);
    out[b +  6] = as_type<uint>(fmin_v2.x);
    out[b +  7] = as_type<uint>(fmin_v2.y);
    out[b +  8] = as_type<uint>(fmax_v2.x);
    out[b +  9] = as_type<uint>(fmax_v2.y);
    out[b + 10] = uint(as_type<ushort>(hmax_v3.x));
    out[b + 11] = uint(as_type<ushort>(hmax_v3.y));
    out[b + 12] = uint(as_type<ushort>(hmax_v3.z));
    out[b + 13] = uint(as_type<ushort>(hmed_v4.x));
    out[b + 14] = uint(as_type<ushort>(hmed_v4.y));
    out[b + 15] = uint(as_type<ushort>(hmed_v4.z));
    out[b + 16] = uint(as_type<ushort>(hmed_v4.w));
    out[b + 17] = uint(as_type<ushort>(smax.x));
    out[b + 18] = uint(as_type<ushort>(smax.y));
    out[b + 19] = m1;
    out[b + 20] = m2;
    out[b + 21] = uint(as_type<ushort>(t1));
    out[b + 22] = uint(as_type<ushort>(t2));
    out[b + 23] = as_type<uint>(tanh_v3.x);
    out[b + 24] = as_type<uint>(tanh_v3.z);
    out[b + 25] = as_type<uint>(tanh_v2.y);
    out[b + 26] = as_type<uint>(log_v2.x);
    out[b + 27] = as_type<uint>(log_v2.y);
    out[b + 28] = as_type<uint>(pw.x);
    out[b + 29] = as_type<uint>(pw.y);
    out[b + 30] = as_type<uint>(pw.z);
    out[b + 31] = as_type<uint>(pw.w);
    out[b + 32] = as_type<uint>(qs2);
    out[b + 33] = as_type<uint>(qs0);
    out[b + 34] = as_type<uint>(qv.x);
    out[b + 35] = as_type<uint>(qv.y);
    out[b + 36] = as_type<uint>(qv.z);
    out[b + 37] = as_type<uint>(qv.w);
}
