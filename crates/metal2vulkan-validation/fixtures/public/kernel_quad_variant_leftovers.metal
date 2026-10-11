// Fifteen `air.quad_*` type variants that the corpus reaches once each and that no authored case
// covered: quad_sum.s.i32, quad_shuffle_xor.v2f32, quad_shuffle_up.f32, quad_shuffle_rotate_down.f32,
// quad_shuffle.{v4f16,v2f32,u.i32}, quad_min.u.i32, quad_max.u.i32 and quad_broadcast.{v2f32,v2f16,
// u.v2i16,u.i32,u.i16,f16}. They share one kernel because each is data movement or an integer
// reduction over four lanes, so none is worth a fixture alone and their arguments do not interact.
//
// Every row is exact by construction. The floats are small integers and dyadic fractions, the
// halves likewise, and quad_sum is asked of int32 -- so no lane's answer depends on a rounding mode
// and a wrong LANE, not a wrong ULP, is what this case can see. The inputs are chosen so that the
// four lanes carry four DIFFERENT values in every operand, which is what makes a swapped lane
// visible: shuffle reads lane 3-tid, xor reads lane tid^1, rotate_down reads lane (tid+1)%4 and
// broadcast reads lane 2.
//
// `quad_shuffle_up(x, 1)` is undefined at lane 0, so the shuffle runs unconditionally at top level
// and only the STORE is guarded -- a shuffle is convergent and must not sit under divergent control
// flow, and a lowering that sank it into the branch is a bug this fixture should catch rather than
// hide. Lane 0 writes the marker 0xABCDABCD instead.
//
// Results are bit-cast into one `uint` buffer -- floats whole, halves and ushorts zero-extended --
// so a single output region carries every width without a conversion that would round.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_quad_variant_leftovers(
    device const float4* fin [[buffer(0)]],
    device const half4* hin [[buffer(1)]],
    device const uint4* uin [[buffer(2)]],
    device const int* iin [[buffer(3)]],
    device uint* out [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]])
{
    float4 f = fin[tid];
    half4 h = hin[tid];
    uint4 u = uin[tid];
    int i = iin[tid];
    ushort lane = ushort(3u - tid);
    device uint* row = out + tid * 23;

    row[0] = as_type<uint>(quad_sum(i));

    float2 x = quad_shuffle_xor(f.xy, 1u);
    row[1] = as_type<uint>(x.x);
    row[2] = as_type<uint>(x.y);

    float up = quad_shuffle_up(f.x, 1u);
    if (tid >= 1) { row[3] = as_type<uint>(up); } else { row[3] = 0xABCDABCDu; }

    row[4] = as_type<uint>(quad_shuffle_rotate_down(f.x, 1u));

    half4 sh = quad_shuffle(h, lane);
    row[5] = uint(as_type<ushort>(sh.x));
    row[6] = uint(as_type<ushort>(sh.y));
    row[7] = uint(as_type<ushort>(sh.z));
    row[8] = uint(as_type<ushort>(sh.w));

    float2 sf = quad_shuffle(f.zw, lane);
    row[9] = as_type<uint>(sf.x);
    row[10] = as_type<uint>(sf.y);

    row[11] = quad_shuffle(u.x, lane);
    row[12] = quad_min(u.y);
    row[13] = quad_max(u.y);

    float2 bf = quad_broadcast(f.xy, ushort(2));
    row[14] = as_type<uint>(bf.x);
    row[15] = as_type<uint>(bf.y);

    half2 bh = quad_broadcast(h.xy, ushort(2));
    row[16] = uint(as_type<ushort>(bh.x));
    row[17] = uint(as_type<ushort>(bh.y));

    ushort2 bs = quad_broadcast(ushort2(u.z, u.w), ushort(2));
    row[18] = uint(bs.x);
    row[19] = uint(bs.y);

    row[20] = quad_broadcast(u.z, ushort(2));
    row[21] = uint(quad_broadcast(ushort(u.w), ushort(2)));
    row[22] = uint(as_type<ushort>(quad_broadcast(h.z, ushort(2))));
}
