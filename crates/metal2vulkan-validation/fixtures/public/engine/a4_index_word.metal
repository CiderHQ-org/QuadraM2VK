// Owned fixture: 64-bit index arithmetic whose only use is a buffer index.
// The compiler keeps `ulong(t) * 3 + 1` as an `i64` multiply and add feeding the load's
// `getelementptr`, which the engine holds as its low word (`arith.index_word`):
// out[t] = in[3t + 1]. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_index_word(device const uint *in [[buffer(0)]],
                          device uint *out [[buffer(1)]],
                          uint t [[thread_position_in_grid]]) {
    out[t] = in[ulong(t) * 3ul + 1ul];
}
