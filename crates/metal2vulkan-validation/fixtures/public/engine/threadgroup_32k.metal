// Owned fixture: a static threadgroup array of exactly 32768 bytes, the whole
// maxComputeSharedMemorySize of MoltenVK M3 and lavapipe. The arena must be the exact sum of the
// Metal allocations (no sink word), so this compiles for moltenvk-m3 and lavapipe.
// With a threadgroup of 1024 threads, thread t writes words t + 1024 k (k = 0..7), covering all
// 8192 words, and reads word 8191 - t after the barrier:
//     out[t] = (8191 - t) * 3
// Loop-free by construction.
#include <metal_stdlib>
using namespace metal;

kernel void threadgroup_32k(device uint *out [[buffer(0)]],
                            uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint scratch[8192];
    scratch[t] = t * 3u;
    scratch[t + 1024u] = (t + 1024u) * 3u;
    scratch[t + 2048u] = (t + 2048u) * 3u;
    scratch[t + 3072u] = (t + 3072u) * 3u;
    scratch[t + 4096u] = (t + 4096u) * 3u;
    scratch[t + 5120u] = (t + 5120u) * 3u;
    scratch[t + 6144u] = (t + 6144u) * 3u;
    scratch[t + 7168u] = (t + 7168u) * 3u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[t] = scratch[8191u - t];
}
