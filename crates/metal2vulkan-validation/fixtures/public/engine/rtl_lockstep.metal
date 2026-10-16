// Owned fixture: threadgroup accesses that only
// a simdgroup's lockstep orders, inside loops each thread leaves after its own number of
// iterations, so no barrier the translation may insert there is uniform in the simdgroup: a device
// that runs a simdgroup in lockstep itself orders them, and every other device takes emulated
// simdgroups. A 64-thread threadgroup (two simdgroups): thread t first holds word t of a
// threadgroup array; each iteration it reads another thread's word of its simdgroup (in another
// 8-lane chunk) and writes its own word from it, with no barrier between. Written for this
// repository; no third-party source. Each loop runs at most a buffer word's count of iterations
// (the first) or 8 (the second).
#include <metal_stdlib>
using namespace metal;

// Thread t runs counts[t] iterations; each reads the word of thread t ^ 16, then writes its own.
// Under lockstep the reads of an iteration all precede its writes. out[t] is thread t's word after
// the loop.
kernel void rtl_lockstep_divergent_loop(device const uint *counts [[buffer(0)]],
                                        device uint *out [[buffer(1)]],
                                        uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint words[64];
    words[t] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = counts[t];
    for (uint i = 0; i < n; ++i) {
        uint x = words[t ^ 16u];
        words[t] = x * 3u + i + 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[t] = words[t];
}

// Thread t, its word t + 1 at first, adds the word of the thread 8 lanes on in its simdgroup to
// its own while its own is below limits[t], at most 8 times: the loop's exit depends on what the
// other threads wrote. out[t] is thread t's word after the loop and out[64 + t] its count.
kernel void rtl_lockstep_until(device const uint *limits [[buffer(0)]],
                               device uint *out [[buffer(1)]],
                               uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint words[64];
    words[t] = t + 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint limit = limits[t];
    uint steps = 0;
    while (words[t] < limit && steps < 8u) {
        uint x = words[((t + 8u) & 31u) | (t & 32u)];
        words[t] = words[t] + x;
        ++steps;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[t] = words[t];
    out[64 + t] = steps;
}
