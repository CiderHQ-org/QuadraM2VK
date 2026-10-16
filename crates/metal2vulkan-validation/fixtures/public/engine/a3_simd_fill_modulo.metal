// Owned fixture: simd_shuffle_and_fill_{down,up} with segments
// (`modulo`) of 4 and 8 lanes, in-segment reads and fills. Every thread writes eight words,
// out[8t + i], t = thread_index_in_threadgroup, from data t and fill t + 1000, so each word names
// the lane and the operand it came from. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a3_simd_fill_modulo(device uint *out [[buffer(0)]],
                                uint t [[thread_index_in_threadgroup]]) {
    uint data = t;
    uint fill = t + 1000u;
    device uint *o = out + 8u * t;
    o[0] = simd_shuffle_and_fill_down(data, fill, ushort(1), ushort(4));
    o[1] = simd_shuffle_and_fill_down(data, fill, ushort(3), ushort(4));
    o[2] = simd_shuffle_and_fill_up(data, fill, ushort(1), ushort(4));
    o[3] = simd_shuffle_and_fill_up(data, fill, ushort(3), ushort(4));
    o[4] = simd_shuffle_and_fill_down(data, fill, ushort(3), ushort(8));
    o[5] = simd_shuffle_and_fill_down(data, fill, ushort(6), ushort(8));
    o[6] = simd_shuffle_and_fill_up(data, fill, ushort(2), ushort(8));
    o[7] = simd_shuffle_and_fill_up(data, fill, ushort(7), ushort(8));
}
