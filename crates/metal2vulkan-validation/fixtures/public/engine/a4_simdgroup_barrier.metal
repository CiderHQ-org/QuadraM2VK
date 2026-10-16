// Owned fixture: a simdgroup barrier orders threadgroup memory among a
// simdgroup's threads (`barrier.simdgroup`). Each thread of a 32-thread threadgroup stores
// 3t + 1, then, after the barrier, reads the word of thread t ^ 1, which is in its own simdgroup
// at any even simdgroup width: out[t] = 3 (t ^ 1) + 1. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_simdgroup_barrier(device uint *out [[buffer(0)]],
                                 uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint words[32];
    words[t] = t * 3u + 1u;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    out[t] = words[t ^ 1u];
}
