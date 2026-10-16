// Owned fixture: an object function whose threadgroup memory arguments the render encoder sizes
// (`setObjectThreadgroupMemoryLength:atIndex:`), beside a static threadgroup array.
// `mesh_object_memory_sums`' threads each store a word of object buffer 0 into
// `[[threadgroup(0)]]`, a `uint4` into `[[threadgroup(1)]]` and a word into the static array; after
// a threadgroup barrier its first 8 threads each combine three neighbours' values, one from each,
// into their payload word, and thread 0 launches one mesh threadgroup. Run under one object
// threadgroup of 16 threads, with lengths of 64 and 256 bytes; a misplaced argument or array
// reads another's values. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ObjectMemoryPayload {
    uint words[8];
};

[[object, max_total_threads_per_threadgroup(16)]]
void mesh_object_memory_sums(object_data ObjectMemoryPayload& payload [[payload]],
                             mesh_grid_properties grid,
                             const device uint* input [[buffer(0)]],
                             uint tid [[thread_index_in_threadgroup]],
                             threadgroup uint* first [[threadgroup(0)]],
                             threadgroup uint4* second [[threadgroup(1)]]) {
    threadgroup uint fixed[16];
    first[tid] = input[tid] + tid;
    second[tid] = uint4(input[tid] * 3u, tid + 7u, 0u, 0u);
    fixed[tid] = 100u + tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
        payload.words[tid] = first[(tid + 1u) & 15u]
            + (second[(tid + 5u) & 15u].x << 8)
            + (second[(tid + 9u) & 15u].y << 16)
            + (fixed[(tid + 3u) & 15u] << 24);
    }
    if (tid == 0u) {
        grid.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}
