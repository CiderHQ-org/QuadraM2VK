// Sixteen `air.simd_*` type variants that the corpus reaches once each and that no authored case
// covered: simd_sum.{v2f32,u.v2i32}, simd_shuffle_xor.{v4f32,u.i16,s.v2i16},
// simd_shuffle_up.{v4f16,u.i8,u.i16}, simd_shuffle_and_fill_down.{v4f32,v3f16},
// simd_prefix_inclusive_sum.f32, simd_prefix_exclusive_sum.u.i16, simd_min.s.v2i16,
// simd_max.s.v2i16, simd_broadcast_first.f32 and simd_all. One dispatch of 32 threads in one
// threadgroup is exactly one Apple simdgroup.
//
// Every reduction here is over integers small enough to be exact in binary32 (and in ushort) at ANY
// association order, so a reassociating compiler cannot move a bit: the float pair sums to
// (528, 1056), the uint pair to (528, 1584), and the inclusive prefix at lane t is (t+1)(t+2)/2.
// That is deliberate -- the point of this case is to see a wrong LANE, not a wrong ULP, and the 32
// lanes therefore carry 32 different values in every operand.
//
// `simd_shuffle_up(x, delta)` is undefined at lanes below `delta`, so all three shuffle_up calls run
// unconditionally at top level and only the STORE is guarded -- a shuffle is convergent and must not
// sit under divergent control flow. The half4 row uses an `if`/`else` and the two integer rows use a
// ternary ON PURPOSE: the ternary is the `OpSelect` shape that SPIRV-Cross sinks a single-reader
// subgroup result into, so those two rows exercise the materialization that stops it, while the
// `if`/`else` rows show the same answers with no select in sight. Guarded lanes write 0xABCDABCD.
//
// `simd_shuffle_and_fill_down` needs no guard: stepping off the end into the filling vector is what
// the family is for, so the tail lanes' answers are fully defined.
//
// Results are bit-cast into one `uint` buffer -- floats whole, halves and ushorts zero-extended --
// so a single output region carries every width without a conversion that would round.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_simd_variant_leftovers(
    device const float4* fa [[buffer(0)]],
    device const float4* fb [[buffer(1)]],
    device const float4* fc [[buffer(2)]],
    device const half4* ha [[buffer(3)]],
    device const half4* hb [[buffer(4)]],
    device const uint4* ua [[buffer(5)]],
    device const int4* ia [[buffer(6)]],
    device uint* out [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]])
{
    float4 c = fa[tid];
    float4 cf = fb[tid];
    float4 m = fc[tid];
    half4 g = ha[tid];
    half4 gf = hb[tid];
    uint4 u = ua[tid];
    int4 i = ia[tid];

    float2 a = m.xy;
    uint2 b = u.xy;
    ushort d = ushort(u.z);
    ushort q = ushort(u.w);
    short2 e = short2(short(i.x), short(i.y));
    uchar e8 = uchar(i.z);

    device uint* row = out + tid * 33;

    float2 sa = simd_sum(a);
    row[0] = as_type<uint>(sa.x);
    row[1] = as_type<uint>(sa.y);

    uint2 sb = simd_sum(b);
    row[2] = sb.x;
    row[3] = sb.y;

    float4 xc = simd_shuffle_xor(c, 8u);
    row[4] = as_type<uint>(xc.x);
    row[5] = as_type<uint>(xc.y);
    row[6] = as_type<uint>(xc.z);
    row[7] = as_type<uint>(xc.w);

    row[8] = uint(simd_shuffle_xor(d, 4u));

    short2 xe = simd_shuffle_xor(e, 16u);
    row[9] = uint(as_type<ushort>(xe.x));
    row[10] = uint(as_type<ushort>(xe.y));

    half4 ug = simd_shuffle_up(g, 3u);
    uchar u8 = simd_shuffle_up(e8, 2u);
    ushort ud = simd_shuffle_up(d, 5u);
    if (tid >= 3) {
        row[11] = uint(as_type<ushort>(ug.x));
        row[12] = uint(as_type<ushort>(ug.y));
        row[13] = uint(as_type<ushort>(ug.z));
        row[14] = uint(as_type<ushort>(ug.w));
    } else {
        row[11] = 0xABCDABCDu; row[12] = 0xABCDABCDu;
        row[13] = 0xABCDABCDu; row[14] = 0xABCDABCDu;
    }
    row[15] = tid >= 2 ? uint(u8) : 0xABCDABCDu;
    row[16] = tid >= 5 ? uint(ud) : 0xABCDABCDu;

    float4 fdc = simd_shuffle_and_fill_down(c, cf, 6u);
    row[17] = as_type<uint>(fdc.x);
    row[18] = as_type<uint>(fdc.y);
    row[19] = as_type<uint>(fdc.z);
    row[20] = as_type<uint>(fdc.w);

    half3 fdg = simd_shuffle_and_fill_down(g.xyz, gf.xyz, 9u);
    row[21] = uint(as_type<ushort>(fdg.x));
    row[22] = uint(as_type<ushort>(fdg.y));
    row[23] = uint(as_type<ushort>(fdg.z));

    row[24] = as_type<uint>(simd_prefix_inclusive_sum(m.z));
    row[25] = uint(simd_prefix_exclusive_sum(q));

    short2 mn = simd_min(e);
    short2 mx = simd_max(e);
    row[26] = uint(as_type<ushort>(mn.x));
    row[27] = uint(as_type<ushort>(mn.y));
    row[28] = uint(as_type<ushort>(mx.x));
    row[29] = uint(as_type<ushort>(mx.y));

    row[30] = as_type<uint>(simd_broadcast_first(m.w));
    row[31] = simd_all(i.w != 0) ? 1u : 0u;
    row[32] = simd_all(i.w >= 0) ? 1u : 0u;
}
