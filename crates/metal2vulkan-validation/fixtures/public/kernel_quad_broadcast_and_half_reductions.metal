// Six uncovered subgroup families in one 32-lane simdgroup: `air.quad_is_first` (6 corpus
// sources), `air.quad_shuffle_xor.v4f16` (5), `air.quad_broadcast.v4f32` (5), `air.simd_sum.v4f16`
// (4), `air.quad_sum.v4f16` (4) and `air.quad_shuffle_down.u.v4i32` (4).
//
// The two reductions are the reason the half lanes carry small INTEGERS. A sum over 32 lanes is
// only well defined for an exact comparison if every partial sum is exact whatever order the two
// runtimes accumulate in; half is exact on integers up to 2048, and the largest column here totals
// 1552, so every association gives the same bits. The lane values still differ per lane and per
// component, so a reduction that summed the wrong set is not hidden.
//
// The reductions also separate the two lane models: `quad_sum` clusters to four lanes and
// `simd_sum` to all thirty-two, so the eight quads must report eight different quad sums and one
// common simd sum. `quad_shuffle_xor` with mask 1 and `quad_broadcast` with index 2 stay inside the
// quad for every lane, so they are defined everywhere; `quad_shuffle_down` with delta 1 leaves the
// last lane of each quad undefined and those slots are overwritten rather than asserted on. The
// branch is on tid, never on a shuffle result -- selecting on the result lets SPIRV-Cross sink the
// shuffle into an MSL ternary arm.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_quad_broadcast_and_half_reductions(
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint d = tid + 1;
    half4 hd = half4(half(d), half(d) + 1.0h, half(d) * 2.0h, half(d) + 32.0h);
    float4 f4 = float4(float(d), float(d) * 2.0f, float(d) * 4.0f, float(d) * 8.0f);
    uint4 u4 = uint4(d, d * 3u, d * 5u, d * 7u);

    half4 xr = quad_shuffle_xor(hd, 1);
    float4 bc = quad_broadcast(f4, 2);
    half4 sums = simd_sum(hd);
    half4 quads = quad_sum(hd);
    uint4 down = quad_shuffle_down(u4, 1);
    bool first = quad_is_first();

    uint b = tid * 18;
    out[b + 0] = uint(as_type<ushort>(xr.x));
    out[b + 1] = uint(as_type<ushort>(xr.y));
    out[b + 2] = uint(as_type<ushort>(xr.z));
    out[b + 3] = uint(as_type<ushort>(xr.w));
    out[b + 4] = as_type<uint>(bc.x);
    out[b + 5] = as_type<uint>(bc.y);
    out[b + 6] = as_type<uint>(bc.z);
    out[b + 7] = as_type<uint>(bc.w);
    out[b + 8] = uint(as_type<ushort>(sums.x));
    out[b + 9] = uint(as_type<ushort>(sums.y));
    out[b + 10] = uint(as_type<ushort>(sums.z));
    out[b + 11] = uint(as_type<ushort>(sums.w));
    out[b + 12] = uint(as_type<ushort>(quads.x));
    out[b + 13] = uint(as_type<ushort>(quads.w));
    out[b + 14] = down.x;
    out[b + 15] = down.y;
    out[b + 16] = down.w;
    out[b + 17] = first ? 1u : 0u;

    // The last lane of each quad has no lane one above it inside the quad, so its
    // quad_shuffle_down result is undefined. Pin a constant there instead.
    if (tid % 4 == 3) {
        out[b + 14] = 0;
        out[b + 15] = 0;
        out[b + 16] = 0;
    }
}
