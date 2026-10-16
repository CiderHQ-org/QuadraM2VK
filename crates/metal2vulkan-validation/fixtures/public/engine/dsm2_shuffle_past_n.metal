// Owned fixture: what a simdgroup shuffle reads from a lane past the
// threadgroup's thread count (Metal leaves the value undefined, and the translator's
// emulated simdgroups read the calling lane's). Every thread runs the shuffles, in
// workgroup-uniform control flow; a threadgroup whose size is not a multiple of 32 leaves its
// last simdgroup partial, and its lanes past the thread count do not exist.
// metal-fast-math: off
//
// Thread g of a grid of n threads (g = thread_position_in_grid, lane L its lane in the
// simdgroup) reads its word v = in[g] and writes three words: out[3g] is simd_shuffle(v, src[L])
// (an absolute source lane per lane), out[3g + 1] simd_shuffle_down(v, src[32]), and out[3g + 2]
// simd_shuffle_xor(v, src[33]) (a delta and a mask equal across the simdgroup, as Metal requires
// of them). Moving a uint moves its bits. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void dsm2_shuffle_past_n(device const uint *in [[buffer(0)]],
                                device const uint *src [[buffer(1)]],
                                device uint *out [[buffer(2)]],
                                uint g [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    uint v = in[g];
    out[3u * g] = simd_shuffle(v, ushort(src[lane]));
    out[3u * g + 1u] = simd_shuffle_down(v, ushort(src[32]));
    out[3u * g + 2u] = simd_shuffle_xor(v, ushort(src[33]));
}
