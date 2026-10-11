// Twenty-three uncovered `air.convert.*` families in one kernel, plus five that come free with
// them: every width and signedness change between 8-, 16-, 32- and 64-bit integers, both
// directions between 16-bit integers and halves, and both boolean widenings.
//
// air.convert.u.v4i16.u.v4i1 (2 corpus sources), air.convert.u.v4i16.f.v4f16 (2),
// air.convert.u.v3i16.u.v3i8 (2), air.convert.s.v2i32.s.v2i8 (2), air.convert.s.v2i16.s.v2i32 (2),
// air.convert.f.v4f16.u.v4i16 (2), and seventeen more at one source each -- 29 sources.
//
// A width conversion has exactly two ways to be wrong, and both are silent: it can sign-extend
// where it should zero-extend, and it can keep bits it should have dropped. Every input here is
// chosen so both show.
//
//   * The narrowing sources overflow their destination on purpose. `t * 300 - 500` and
//     `-t * 70000 + 5` do not fit in 8 or 16 bits, so `char4(i4)` and `short4(i4)` must drop the
//     high bits rather than clamp, and the surviving low bits differ per thread.
//   * Half the lanes are negative. `t - 3` is negative for the first three threads and `-t * 70000`
//     for all but the first, so a widening that zero-extended a signed source, or a narrowing read
//     back as signed when the symbol says unsigned, lands a number four orders of magnitude away
//     rather than a plausible one. `u4.w` is 4294967040 + t, which is negative as an int32 and
//     enormous as a uint32 -- the same disagreement at the other end.
//   * The halves are all non-negative and below 256 with a fraction of 3/4 or 1/2, so the
//     float-to-integer direction is exact and truncates toward zero rather than rounding, and the
//     integer-to-float direction is exact for every value it is given.
//   * The 64-bit results are written as two 32-bit words each, low first, so a conversion that
//     produced the right low word and the wrong sign extension is still visible.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_integer_width_conversions(
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    int t = int(tid);
    int4   i4  = int4(t - 3, t * 300 - 500, -t * 70000 + 5, t * 7);
    uint4  u4  = uint4(uint(t) + 1u, uint(t) * 300u, uint(t) * 70000u, 4294967040u + uint(t));
    short4 s4  = short4(short(t) - 4, short(t) * 100, short(t) * -300, short(t) + 9);
    char4  c4  = char4(char(t) - 3, char(t) * 5 - 60, char(t) * -7, char(t) + 1);
    uchar3 uc3 = uchar3(uchar(t), uchar(t) * 30, uchar(200 + t));
    half4  h4  = half4(half(t) + 0.75h, half(t) * 2.0h + 0.5h, half(t) * 8.0h, half(200 + t));
    half3  h3  = h4.xyz;
    long4  l4  = long4(i4) * 1000000L;
    ulong4 ul4 = ulong4(u4) * 1000UL;
    bool4  b4  = i4 > 0;
    bool2  b2  = b4.xy;

    ushort4 r0 = ushort4(b4);
    uint4   r1 = uint4(ushort4(h4));
    ushort3 r2 = ushort3(uc3);
    int2    r3 = int2(c4.xy);
    short2  r4 = short2(i4.xy);
    half4   r5 = half4(ushort4(u4));
    uchar4  r6 = uchar4(i4);
    ulong4  r7 = ulong4(u4);
    ulong4  r8 = ulong4(i4);
    uchar3  r9 = uchar3(h3);
    uint3   ra = uint3(uc3);
    ushort3 rb = ushort3(h3);
    ulong2  rc = ulong2(u4.xy);
    char4   rd = char4(u4);
    char4   re = char4(i4);
    int4    rf = int4(uchar4(u4));
    int4    rg = int4(ul4);
    int4    rh = int4(c4);
    int4    ri = int4(l4);
    int4    rj = int4(s4);
    short4  rk = short4(i4);
    short2  rl = short2(b2);
    half4   rm = half4(uint4(uint(t) + 1u, 3u * uint(t), 5u * uint(t) + 2u, 7u * uint(t) + 9u));

    uint b = tid * 90u;
    out[b + 0] = uint(r0.x); out[b + 1] = uint(r0.y); out[b + 2] = uint(r0.z);
    out[b + 3] = uint(r0.w);
    out[b + 4] = r1.x; out[b + 5] = r1.y; out[b + 6] = r1.z; out[b + 7] = r1.w;
    out[b + 8] = uint(r2.x); out[b + 9] = uint(r2.y); out[b + 10] = uint(r2.z);
    out[b + 11] = uint(r3.x); out[b + 12] = uint(r3.y);
    out[b + 13] = uint(as_type<ushort>(r4.x)); out[b + 14] = uint(as_type<ushort>(r4.y));
    out[b + 15] = uint(as_type<ushort>(r5.x)); out[b + 16] = uint(as_type<ushort>(r5.y));
    out[b + 17] = uint(as_type<ushort>(r5.z)); out[b + 18] = uint(as_type<ushort>(r5.w));
    out[b + 19] = uint(r6.x); out[b + 20] = uint(r6.y); out[b + 21] = uint(r6.z);
    out[b + 22] = uint(r6.w);
    out[b + 23] = uint(r7.x & 0xFFFFFFFFUL); out[b + 24] = uint(r7.x >> 32);
    out[b + 25] = uint(r7.y & 0xFFFFFFFFUL); out[b + 26] = uint(r7.y >> 32);
    out[b + 27] = uint(r7.z & 0xFFFFFFFFUL); out[b + 28] = uint(r7.z >> 32);
    out[b + 29] = uint(r7.w & 0xFFFFFFFFUL); out[b + 30] = uint(r7.w >> 32);
    out[b + 31] = uint(ulong(r8.x) & 0xFFFFFFFFUL); out[b + 32] = uint(ulong(r8.x) >> 32);
    out[b + 33] = uint(ulong(r8.y) & 0xFFFFFFFFUL); out[b + 34] = uint(ulong(r8.y) >> 32);
    out[b + 35] = uint(ulong(r8.z) & 0xFFFFFFFFUL); out[b + 36] = uint(ulong(r8.z) >> 32);
    out[b + 37] = uint(ulong(r8.w) & 0xFFFFFFFFUL); out[b + 38] = uint(ulong(r8.w) >> 32);
    out[b + 39] = uint(r9.x); out[b + 40] = uint(r9.y); out[b + 41] = uint(r9.z);
    out[b + 42] = ra.x; out[b + 43] = ra.y; out[b + 44] = ra.z;
    out[b + 45] = uint(rb.x); out[b + 46] = uint(rb.y); out[b + 47] = uint(rb.z);
    out[b + 48] = uint(rc.x & 0xFFFFFFFFUL); out[b + 49] = uint(rc.x >> 32);
    out[b + 50] = uint(rc.y & 0xFFFFFFFFUL); out[b + 51] = uint(rc.y >> 32);
    out[b + 52] = uint(as_type<uchar>(rd.x)); out[b + 53] = uint(as_type<uchar>(rd.y));
    out[b + 54] = uint(as_type<uchar>(rd.z)); out[b + 55] = uint(as_type<uchar>(rd.w));
    out[b + 56] = uint(as_type<uchar>(re.x)); out[b + 57] = uint(as_type<uchar>(re.y));
    out[b + 58] = uint(as_type<uchar>(re.z)); out[b + 59] = uint(as_type<uchar>(re.w));
    out[b + 60] = uint(rf.x); out[b + 61] = uint(rf.y); out[b + 62] = uint(rf.z);
    out[b + 63] = uint(rf.w);
    out[b + 64] = uint(rg.x); out[b + 65] = uint(rg.y); out[b + 66] = uint(rg.z);
    out[b + 67] = uint(rg.w);
    out[b + 68] = uint(rh.x); out[b + 69] = uint(rh.y); out[b + 70] = uint(rh.z);
    out[b + 71] = uint(rh.w);
    out[b + 72] = uint(ri.x); out[b + 73] = uint(ri.y); out[b + 74] = uint(ri.z);
    out[b + 75] = uint(ri.w);
    out[b + 76] = uint(rj.x); out[b + 77] = uint(rj.y); out[b + 78] = uint(rj.z);
    out[b + 79] = uint(rj.w);
    out[b + 80] = uint(as_type<ushort>(rk.x)); out[b + 81] = uint(as_type<ushort>(rk.y));
    out[b + 82] = uint(as_type<ushort>(rk.z)); out[b + 83] = uint(as_type<ushort>(rk.w));
    out[b + 84] = uint(as_type<ushort>(rl.x)); out[b + 85] = uint(as_type<ushort>(rl.y));
    out[b + 86] = uint(as_type<ushort>(rm.x)); out[b + 87] = uint(as_type<ushort>(rm.y));
    out[b + 88] = uint(as_type<ushort>(rm.z)); out[b + 89] = uint(as_type<ushort>(rm.w));
}
