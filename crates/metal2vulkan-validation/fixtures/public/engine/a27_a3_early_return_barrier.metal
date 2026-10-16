// metal-std: metal4.0
// metal-macos-min: 27.0
// Owned fixture: the AIR 2.9 twin of
// validation/fixtures/public/engine/a3_early_return_barrier.metal. Below the next blank line is
// that file, unchanged; the directives above compile it as MSL 4.0 for macOS 27, where Apple's
// compiler writes AIR 2.9 (each memory atomic with its `mem_flags`, each barrier with its
// `memory_order`). It compiles on every committed profile to the same words as its original
// (tests/engine_air_2_9.rs), so the original's cases and rows speak for it.

// Owned fixture: threads return before a threadgroup
// barrier and between two barriers; the threads that remain exchange words through threadgroup
// memory across each barrier. Threads t >= limits[0] return before the first barrier, threads
// t >= limits[1] (limits[1] <= limits[0]) between the two; each thread writes out[t] once, on the
// path it takes. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a3_early_return_barrier(device const uint *limits [[buffer(0)]],
                                    device uint *out [[buffer(1)]],
                                    uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint a[128];
    threadgroup uint b[128];
    uint first = limits[0];
    uint second = limits[1];
    if (t >= first) {
        out[t] = 0xE0000000u | t;
        return;
    }
    a[t] = t * 3u + 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint left = a[(t + 1u) % first];
    if (t >= second) {
        out[t] = 0xE1000000u | left;
        return;
    }
    b[t] = left + 1000u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[t] = b[(t + 5u) % second];
}
