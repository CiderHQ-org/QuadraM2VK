// Thirteen uncovered scalar and vector families in one kernel, every one anchored where its answer
// is exact.
//
// `air.clamp.f32` (3 corpus sources), `air.clamp.u.v4i32` (3), `air.clamp.s.v3i16` (3),
// `air.fabs.f32` (3), `air.min.u.v3i16` (3), `air.fmod.f16` (3), `air.fast_log.v2f32` (3),
// `air.fast_rsqrt.v2f32` (3), `air.fast_sinpi.f32` (3), `air.fmod.f32` (1), `air.fmod.v3f16` (1),
// `air.min.s.v2i16` (1), `air.min.s.v3i32` (1), `air.min.u.v3i32` (1) -- 32 sources.
//
// The `precise::` prefixes are not decoration. Under Metal's default fast math `fabs`, `clamp` and
// a float `fmod` compile to `air.fast_*`, which are different symbols that are already covered;
// `precise::` is what reaches the plain ones these sources call. At half width there is no fast
// variant, so `fmod` on `half` and `half3` needs no prefix.
//
// Every input is chosen so the answer is forced rather than approximated:
//
//  - Each clamp and min sees one operand below its bound, one inside and one above, so a lowering
//    that dropped a bound, swapped the two, or confused signed with unsigned lands on a different
//    word for at least one lane. The unsigned lanes carry values above 2^31 and the signed ones
//    carry negatives, which is the pair that separates the two comparisons.
//  - `fmod` is exact whenever its operands are: the remainder of two representable values is
//    representable. `fmod(-5.5, 4)` is -1.5, which takes the sign of the dividend, so a lowering
//    that returned a Euclidean remainder differs.
//  - `fast_rsqrt` is NOT correctly rounded on this host, so it is asked only at powers of four,
//    where its answer is a power of two and no approximation has room to differ.
//  - `fast_sinpi` returns a signed zero at integers, so it is asked at a half-integer, where the
//    answer is exactly -1.
//  - `fast_log` is exactly zero at one; its second lane is at ten, which is not derivable exactly
//    and is checked against a float64 derivation to within one ULP instead. Zero alone would not
//    separate this from a base-two or base-ten logarithm, which is what the second lane is for.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_precise_clamp_min_fmod_and_fast_roots(
    device uint *out [[buffer(0)]],
    device const uint *in [[buffer(1)]])
{
    float   f0 = as_type<float>(in[0]);
    float   f1 = as_type<float>(in[1]);
    uint4   u4 = uint4(in[2], in[3], in[4], in[5]);
    short3  s3 = short3(short(in[6]), short(in[7]), short(in[8]));
    ushort3 h3 = ushort3(ushort(in[9]), ushort(in[10]), ushort(in[11]));
    short2  s2 = short2(short(in[12]), short(in[13]));
    int3    i3 = int3(int(in[14]), int(in[15]), int(in[16]));
    uint3   q3 = uint3(in[17], in[18], in[19]);
    half    hf = half(as_type<float>(in[20]));
    half3  h3f = half3(half(as_type<float>(in[21])),
                       half(as_type<float>(in[22])),
                       half(as_type<float>(in[23])));
    float   f2 = as_type<float>(in[24]);
    float2  r2 = float2(as_type<float>(in[25]), as_type<float>(in[26]));
    float2  l2 = float2(as_type<float>(in[27]), as_type<float>(in[28]));
    float   sp = as_type<float>(in[29]);

    out[0] = as_type<uint>(precise::fabs(f0));
    out[1] = as_type<uint>(precise::clamp(f1, 1.0f, 2.0f));
    uint4 cu = clamp(u4, uint4(2u), uint4(5u));
    out[2] = cu.x; out[3] = cu.y; out[4] = cu.z; out[5] = cu.w;
    short3 cs = clamp(s3, short3(-3), short3(3));
    out[6] = uint(int(cs.x)); out[7] = uint(int(cs.y)); out[8] = uint(int(cs.z));
    ushort3 mn = min(h3, ushort3(4));
    out[9] = uint(mn.x); out[10] = uint(mn.y); out[11] = uint(mn.z);
    short2 ms = min(s2, short2(1));
    out[12] = uint(int(ms.x)); out[13] = uint(int(ms.y));
    int3 mi = min(i3, int3(7));
    out[14] = uint(mi.x); out[15] = uint(mi.y); out[16] = uint(mi.z);
    uint3 mq = min(q3, uint3(6u));
    out[17] = mq.x; out[18] = mq.y; out[19] = mq.z;
    out[20] = uint(as_type<ushort>(fmod(hf, 2.0h)));
    half3 hm = fmod(h3f, half3(4.0h));
    out[21] = uint(as_type<ushort>(hm.x));
    out[22] = uint(as_type<ushort>(hm.y));
    out[23] = uint(as_type<ushort>(hm.z));
    out[24] = as_type<uint>(precise::fmod(f2, 4.0f));
    float2 rr = fast::rsqrt(r2);
    out[25] = as_type<uint>(rr.x); out[26] = as_type<uint>(rr.y);
    float2 ll = fast::log(l2);
    out[27] = as_type<uint>(ll.x); out[28] = as_type<uint>(ll.y);
    out[29] = as_type<uint>(fast::sinpi(sp));
}
