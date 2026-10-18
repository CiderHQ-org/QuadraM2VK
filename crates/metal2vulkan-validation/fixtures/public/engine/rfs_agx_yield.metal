// air-rename: _Z9rfs_yieldt llvm.agx3.yield
// Owned fixture: `llvm.agx3.yield`, the backoff Apple's decoupled-lookback scans call
// in their spin loops. MSL has no spelling for it, so the kernels call `rfs_yield`, which this
// source declares and never defines, and the directive above renames every call to the
// intrinsic in both companions (scripts/regen-air-fixtures). Measured on an M3 Max and an M4: the
// yield delays the calling thread for a time that grows with its operand and changes no word, with
// or without Metal's validation layers.
#include <metal_stdlib>
using namespace metal;

void rfs_yield(ushort n);

// Yields with constant and run-time operands between arithmetic, device stores and loads. Words
// [0, 64) are the hash, [64, 128) the reloaded hash plus one XOR the hash, [128, 192) n + i.
kernel void agx_yield_between_accesses(device uint *out [[buffer(0)]],
                                       constant uint &n [[buffer(1)]],
                                       uint i [[thread_position_in_grid]]) {
    uint v = i * 2654435761u;
    rfs_yield(0);
    out[i] = v;
    rfs_yield(ushort(n));
    uint w = out[i] + 1u;
    rfs_yield(ushort(i));
    out[i + 64] = w ^ v;
    rfs_yield(65535);
    out[i + 128] = n + i;
}

// A decoupled-lookback chain: each threadgroup takes a ticket, and its lane 0 spins, yielding a
// growing backoff, until its predecessor's word ((prefix << 1) | 1) is published, then publishes
// its own inclusive prefix of ticket + 1. A group that spins `cap` times gives up and publishes
// nothing, so every loop is bounded and a group that waited too long leaves its word, and its
// successors', zero.
kernel void agx_yield_lookback(device atomic_uint *ticket [[buffer(0)]],
                               device atomic_uint *state [[buffer(1)]],
                               constant uint &cap [[buffer(2)]],
                               uint lid [[thread_index_in_threadgroup]]) {
    threadgroup uint tg;
    if (lid == 0) {
        tg = atomic_fetch_add_explicit(ticket, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint g = tg;
    if (lid != 0) {
        return;
    }
    uint prev = 0;
    if (g > 0) {
        uint k = 0;
        uint s = atomic_load_explicit(&state[g - 1], memory_order_relaxed);
        while ((s & 1u) == 0u) {
            k++;
            if (k >= cap) {
                return;
            }
            rfs_yield(ushort(min(k, 64u)));
            s = atomic_load_explicit(&state[g - 1], memory_order_relaxed);
        }
        prev = s >> 1;
    }
    atomic_store_explicit(&state[g], ((prev + g + 1u) << 1) | 1u, memory_order_relaxed);
}
