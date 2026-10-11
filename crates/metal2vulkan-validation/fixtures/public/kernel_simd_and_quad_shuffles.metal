// Ten uncovered subgroup and quad shuffle families over one 32-lane threadgroup.
//
// `air.simd_shuffle_xor.v4f16` (2 corpus sources), `air.simd_shuffle_xor.v2f16` (2),
// `air.simd_shuffle_up.v3f16` (2), `air.simd_shuffle_down.u.v4i16` (2),
// `air.simd_shuffle_and_fill_down.v2f16` (2), `air.simd_prefix_exclusive_sum.v4f32` (2),
// `air.simd_xor.u.v2i16` (2), `air.quad_shuffle_xor.f16` (2), `air.quad_shuffle_xor.v2f16` (2) and
// `air.quad_shuffle_xor.v3f16` (2) -- 20 sources.
//
// A shuffle is only evidence if the lane it reads is not the lane it runs on, so every source value
// is a function of `tid` and every read lands somewhere else: an xor by 1, 2 or 3, a step up by one
// lane, a step down by one. And it is only evidence about the COMPONENT if the components differ,
// so each vector's lanes are scaled apart -- a lowering that shuffled component 0 and broadcast it,
// or that swizzled, lands a value the derivation places on another component.
//
// `simd_shuffle_up` at lane 0 and `simd_shuffle_down` at lane 31 read a lane that does not exist,
// and Metal leaves that undefined. Recording whatever came back would make the case an assertion
// about undefined behaviour, so those two lanes store a sentinel instead. The shuffles themselves
// still run on every lane -- only the store is guarded -- because a shuffle is convergent and must
// not be put under divergent control flow, and because a lowering that sank the shuffle into the
// branch is a bug this fixture must not hide.
//
// `simd_shuffle_and_fill_down` needs no such guard: that is the whole point of the family. Lane 31
// steps off the end into the filling vector, so its answer is the fill value and nothing else.
//
// The exclusive prefix sum is stored with one added to it. At lane 0 an exclusive scan IS the
// identity, and Metal returns -0.0 for the float one where MoltenVK returns +0.0 -- the signed-zero
// divergence measured across this whole family, and not something a case can assert. Adding one
// lands both on exactly 1.0 and leaves every other lane's value, i * (i - 1) + 1, exact and still
// different from the inclusive scan.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_simd_and_quad_shuffles(
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    half    t  = half(tid);
    half4   a4 = half4(t, t + 1.0h, t + 2.0h, t + 3.0h);
    half2   a2 = half2(t, t + 16.0h);
    half3   a3 = half3(t, t * 2.0h, t * 4.0h);
    ushort4 u4 = ushort4(ushort(tid), ushort(tid + 100), ushort(tid + 200), ushort(tid + 300));
    ushort2 x2 = ushort2(ushort(tid + 1), ushort((tid + 1) * 7));
    float4  p4 = float4(float(tid), float(tid) * 2.0f, 1.0f, 2.0f);

    half4   sx4 = simd_shuffle_xor(a4, 1);
    half2   sx2 = simd_shuffle_xor(a2, 2);
    half3   su3 = simd_shuffle_up(a3, 1);
    ushort4 sd4 = simd_shuffle_down(u4, 1);
    half2   sf2 = simd_shuffle_and_fill_down(a2, half2(99.0h, 99.0h), 1);
    float4  pe4 = simd_prefix_exclusive_sum(p4);
    ushort2 xr  = simd_xor(x2);
    half    qx1 = quad_shuffle_xor(t, 1);
    half2   qx2 = quad_shuffle_xor(a2, 2);
    half3   qx3 = quad_shuffle_xor(a3, 3);

    uint b = tid * 10;
    out[b + 0] = uint(as_type<ushort>(sx4.w));
    out[b + 1] = uint(as_type<ushort>(sx2.y));
    if (tid >= 1) {
        out[b + 2] = uint(as_type<ushort>(su3.z));
    } else {
        out[b + 2] = 0xABCDu;
    }
    if (tid + 1 < 32) {
        out[b + 3] = uint(sd4.z);
    } else {
        out[b + 3] = 0xABCDu;
    }
    out[b + 4] = uint(as_type<ushort>(sf2.y));
    out[b + 5] = as_type<uint>(pe4.y + 1.0f);
    out[b + 6] = uint(xr.x);
    out[b + 7] = uint(as_type<ushort>(qx1));
    out[b + 8] = uint(as_type<ushort>(qx2.y));
    out[b + 9] = uint(as_type<ushort>(qx3.z));
}
