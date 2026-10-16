// Owned fixture: a tangled operation in the tail of a
// multi-level break. Lane t searches a nested loop (trip counts trips[0] x trips[1], authored
// inputs) for its flat stop iteration stop[t]; at it, the tail records which lanes run it together
// (simd_or of the lane bits) and leaves both loops by returning from the inlined helper. out[2t] is
// that mask (0 when the lane never stops), out[2t + 1] = i << 8 | j at the stop (0xFFFF when the
// lane never stops). MSL has no goto, so the break is a return from an inlined helper; the front
// end compiles it with one join per loop. Cyclic: bounded by the authored trip counts.
#include <metal_stdlib>
using namespace metal;

static uint search(uint s, uint outer, uint inner, uint lane, thread uint &mask) {
    uint i = 0u;
    do {
        uint j = 0u;
        do {
            if (i * inner + j == s) {
                mask = simd_or(1u << lane);
                return (i << 8) | j;
            }
            ++j;
        } while (j < inner);
        ++i;
    } while (i < outer);
    return 0xFFFFu;
}

kernel void a3_multi_level_break_tail(device const uint *stop [[buffer(0)]],
                                      device const uint *trips [[buffer(1)]],
                                      device uint *out [[buffer(2)]],
                                      uint t [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    uint mask = 0u;
    uint r = search(stop[t], trips[0], trips[1], lane, mask);
    out[2u * t] = mask;
    out[2u * t + 1u] = r;
}
