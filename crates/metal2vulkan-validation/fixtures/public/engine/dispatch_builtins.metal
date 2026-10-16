// Owned fixture: kernel dispatch builtins, barriers, and fences.
// Each marker kernel reads one scalar builtin only as a switch selector and stores a constant to
// out[0] for the threads whose builtin matches the first case (the second case never matches),
// so its lowering needs no arithmetic. Threads that store, store the same value. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void thread_index_marker(device uint *out [[buffer(0)]],
                                uint t [[thread_index_in_threadgroup]]) {
    switch (t) {
        case 5: out[0] = 105; break;
        case 1000000: out[0] = 7; break;
    }
}

kernel void grid_position_marker(device uint *out [[buffer(0)]],
                                 uint t [[thread_position_in_grid]]) {
    switch (t) {
        case 37: out[0] = 137; break;
        case 1000000: out[0] = 7; break;
    }
}

kernel void group_position_marker(device uint *out [[buffer(0)]],
                                  uint g [[threadgroup_position_in_grid]]) {
    switch (g) {
        case 2: out[0] = 202; break;
        case 1000000: out[0] = 7; break;
    }
}

kernel void threads_per_grid_marker(device uint *out [[buffer(0)]],
                                    uint n [[threads_per_grid]]) {
    switch (n) {
        case 40: out[0] = 140; break;
        case 1000000: out[0] = 7; break;
    }
}

kernel void groups_per_grid_marker(device uint *out [[buffer(0)]],
                                   uint n [[threadgroups_per_grid]]) {
    switch (n) {
        case 3: out[0] = 303; break;
        case 1000000: out[0] = 7; break;
    }
}

kernel void simdgroup_index_marker(device uint *out [[buffer(0)]],
                                   uint s [[simdgroup_index_in_threadgroup]]) {
    switch (s) {
        case 1: out[0] = 301; break;
        case 1000000: out[0] = 7; break;
    }
}

kernel void threadgroup_barrier_store(device uint *out [[buffer(0)]]) {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    out[0] = 42;
}

kernel void simdgroup_barrier_marker(device uint *out [[buffer(0)]],
                                     uint t [[thread_index_in_threadgroup]]) {
    simdgroup_barrier(mem_flags::mem_none);
    switch (t) {
        case 37: out[0] = 137; break;
        case 1000000: out[0] = 7; break;
    }
}

kernel void device_fence_store(device uint *out [[buffer(0)]]) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    out[0] = 42;
}
