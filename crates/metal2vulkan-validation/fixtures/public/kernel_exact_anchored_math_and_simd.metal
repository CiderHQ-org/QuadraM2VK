// Six uncovered families across three groups, all authored on inputs whose answers are exactly
// representable, so the case is a derivation and not a recording of whatever the GPU said:
// `air.rsqrt.v3f32` (4 corpus sources), `air.log2.v3f16` (4), `air.cospi.f16` (4),
// `air.simd_sum.v4f32` (3) and `air.simd_shuffle_up.s.i16` (3).
//
// Why these inputs. Every rsqrt argument is a power of four, so its inverse square root is a power
// of two and correctly rounded means exact -- that matters because the fast_ roots on this host are
// NOT correctly rounded, and an exactly representable ANSWER is not by itself enough to pin one
// (`fast_sqrt(15129)` misses 123). Every log2 argument is a power of two, so the result is a small
// integer, exact in half. cospi is asked only at INTEGERS, where the answer is exactly +-1: at odd
// half-integers Metal returns a signed zero whose sign comes from its argument reduction (`cospi(0.5)`
// is -0) and MoltenVK returns +0 there, so those arguments are not authorable and are left out.
//
// simd_sum is over 32 lanes, so it is association-proof by construction here: every lane
// contributes an integer and the column totals stay far below 2^24, where float is exact on
// integers, so any accumulation order gives the same bits. simd_shuffle_up with delta 3 leaves the
// three lowest lanes of the simdgroup undefined; those slots are overwritten on a branch over tid,
// never over the shuffle result, which would let SPIRV-Cross sink the shuffle into a ternary arm.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_exact_anchored_math_and_simd(
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_threadgroup]])
{
    float s = float(1u << (2u * (tid % 6u)));
    float3 r = precise::rsqrt(float3(4.0f * s, 16.0f * s, 0.25f * s));
    half3 l = log2(half3(half(1u << (tid % 11u)), 0.5h, 1024.0h));
    half c = cospi(half(tid % 8u));
    float4 v = float4(float(tid + 1), 1.0f, float(tid % 4u), 3.0f);
    float4 sv = simd_sum(v);
    short sh = short(int(tid) * 300 - 4000);
    short su = simd_shuffle_up(sh, 3);

    uint b = tid * 9;
    out[b + 0] = as_type<uint>(r.x);
    out[b + 1] = as_type<uint>(r.y);
    out[b + 2] = as_type<uint>(r.z);
    out[b + 3] = uint(as_type<ushort>(l.x));
    out[b + 4] = uint(as_type<ushort>(l.y));
    out[b + 5] = uint(as_type<ushort>(c));
    out[b + 6] = as_type<uint>(sv.x);
    out[b + 7] = as_type<uint>(sv.z);
    out[b + 8] = as_type<uint>(int(su));

    // The three lowest lanes have no lane three below them, so their shuffle_up is undefined.
    if (tid < 3) {
        out[b + 8] = 0;
    }
}
