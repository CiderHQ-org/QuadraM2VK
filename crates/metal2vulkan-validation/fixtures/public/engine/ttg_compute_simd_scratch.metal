// Owned fixture: a compute kernel whose static threadgroup memory
// (a float4 array of 1536, 24 KiB) and a float4 simdgroup shuffle leave a device that emulates
// 32-lane simdgroups in 32 KiB of shared memory fewer threads a threadgroup than Metal's 1024:
// each emulated simdgroup takes 512 scratch bytes beside the array, so 16 fit (512 threads).
// Thread t of threadgroup g, of `size` threads, stages float4(t, g, 0, 1), and after the barrier
// reads the staged value of thread size - 1 - t and shuffles it down one lane, filling from
// float4(lane):
//     out[g * size + t] = lane < 31 ? float4(size - 2 - t, g, 0, 1) : float4(0)
// for threadgroups of whole simdgroups. Thread 0 also writes the array's last element, so the
// whole array is live. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void ttg_compute_simd_scratch(device float4 *out [[buffer(0)]],
                                     uint t [[thread_index_in_threadgroup]],
                                     uint g [[threadgroup_position_in_grid]],
                                     uint size [[threads_per_threadgroup]],
                                     ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 stage[1536];
    stage[t] = float4(float(t), float(g), 0.0f, 1.0f);
    if (t == 0u) {
        stage[1535] = float4(-1.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 here = stage[size - 1u - t];
    out[g * size + t] = simd_shuffle_and_fill_down(here, float4(float(lane)), ushort(1));
}
