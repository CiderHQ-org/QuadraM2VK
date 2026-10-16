// metal-std: metal4.0
// metal-macos-min: 27.0
// Owned fixture: the AIR 2.9 twin of
// validation/fixtures/public/engine/a4_simdgroup_barrier.metal. Below the next blank line is
// that file, unchanged; the directives above compile it as MSL 4.0 for macOS 27, where Apple's
// compiler writes AIR 2.9 (each memory atomic with its `mem_flags`, each barrier with its
// `memory_order`). It compiles on every committed profile to the same words as its original
// (tests/engine_air_2_9.rs), so the original's cases and rows speak for it.

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
