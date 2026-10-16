// Owned fixture: what the other lane-moving simdgroup and quadgroup
// operations read from a lane past the threadgroup's thread count, beside dsm2_shuffle_past_n's
// simd_shuffle, simd_shuffle_down, and simd_shuffle_xor (which read all-zero bits there). Every
// thread runs every operation, in workgroup-uniform control flow; a threadgroup whose size is not
// a multiple of 32 leaves its last simdgroup partial, and its lanes past the thread count do not
// exist.
// metal-fast-math: off
//
// Thread g of a grid of n threads (lane L its lane in the simdgroup) reads its word v = in[g] and
// writes ten words, out[10g + i]:
//   0 simd_broadcast(v, src[32])
//   1 simd_shuffle_rotate_down(v, src[33])
//   2 simd_shuffle_and_fill_up(v, ~v, src[33])             (modulo: the simdgroup's 32 lanes)
//   3 simd_shuffle_and_fill_down(v, ~v, src[33], src[34])
//   4 quad_shuffle_xor(v, src[35])
//   5 quad_broadcast(v, src[36])
//   6 quad_shuffle_down(v, src[37])
//   7 quad_shuffle_rotate_down(v, src[37])
//   8 simd_shuffle of v's bits as a float, from lane src[L]
//   9 simd_shuffle of v's high 16 bits as a half, from lane src[L], zero-extended
// The lane, delta, mask, and modulo operands are equal across the simdgroup, as Metal requires,
// except the per-lane source lanes src[L]. Moving a value moves its bits. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void dsm2_shuffle_past_n_more(device const uint *in [[buffer(0)]],
                                     device const uint *src [[buffer(1)]],
                                     device uint *out [[buffer(2)]],
                                     uint g [[thread_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    uint v = in[g];
    uint w = ~v;
    ushort delta = ushort(src[33]);
    uint o = 10u * g;
    out[o] = simd_broadcast(v, ushort(src[32]));
    out[o + 1u] = simd_shuffle_rotate_down(v, delta);
    out[o + 2u] = simd_shuffle_and_fill_up(v, w, delta);
    out[o + 3u] = simd_shuffle_and_fill_down(v, w, delta, ushort(src[34]));
    out[o + 4u] = quad_shuffle_xor(v, ushort(src[35]));
    out[o + 5u] = quad_broadcast(v, ushort(src[36]));
    out[o + 6u] = quad_shuffle_down(v, ushort(src[37]));
    out[o + 7u] = quad_shuffle_rotate_down(v, ushort(src[37]));
    float f = simd_shuffle(as_type<float>(v), ushort(src[lane]));
    out[o + 8u] = as_type<uint>(f);
    half h = simd_shuffle(as_type<half>(ushort(v >> 16)), ushort(src[lane]));
    out[o + 9u] = uint(as_type<ushort>(h));
}
