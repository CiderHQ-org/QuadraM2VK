// Owned fixture: simdgroup and quad operations in divergent control
// flow, which a kernel whose simdgroups are emulated through workgroup memory runs predicated.
// Every kernel writes out[t], t = thread_index_in_threadgroup, unless it returned first (out[t]
// then keeps its initial 0); lane = t & 31 (Metal simdgroup k is the linear threads
// [32k, 32k + 32)). A lane that does not execute an operation (it returned, took another arm,
// left the loop, or continued past it) is inactive for it, as are the lanes past the
// threadgroup's end. `limit` bounds the threads that do not return early. The expected values are
// derived from t, limit, and the threadgroup size in tests/engine_simd_emulated.rs.
#include <metal_stdlib>
using namespace metal;

// An exit guard before reductions, is_first, and broadcast_first.
kernel void guard_k(device uint *out [[buffer(0)]],
                    constant uint &limit [[buffer(1)]],
                    uint t [[thread_index_in_threadgroup]]) {
    if (t >= limit) {
        return;
    }
    out[t] = simd_sum(t) + uint(simd_is_first()) * 100000u + simd_broadcast_first(t) * 1000000u;
}

// An exit guard before a relative shuffle, a prefix sum, and a vote. A source lane that
// returned is inactive: its value is undefined in Metal, and the emulation reads the caller's.
kernel void guard_shuffle_k(device uint *out [[buffer(0)]],
                            constant uint &limit [[buffer(1)]],
                            uint t [[thread_index_in_threadgroup]]) {
    if (t >= limit) {
        return;
    }
    out[t] = simd_shuffle_down(t, ushort(1)) + simd_prefix_inclusive_sum(1u) * 1000u
        + uint(simd_all(t < 20u)) * 100000u;
}

// Operations in both arms of a divergent branch.
kernel void branch_k(device uint *out [[buffer(0)]],
                     uint t [[thread_index_in_threadgroup]]) {
    uint r;
    if ((t % 3u) == 1u) {
        r = simd_sum(t);
    } else {
        r = simd_max(t) * 1000u + simd_prefix_exclusive_sum(1u);
    }
    out[t] = r;
}

// Nested divergent branches.
kernel void nested_k(device uint *out [[buffer(0)]],
                     uint t [[thread_index_in_threadgroup]]) {
    uint r = 5u;
    if ((t & 1u) != 0u) {
        r = simd_min(t);
        if ((t & 2u) != 0u) {
            r += simd_sum(1u) * 1000u;
        }
    }
    out[t] = r;
}

// A loop whose trip count is the thread's: iteration i runs on the lanes still in the loop.
kernel void trip_k(device uint *out [[buffer(0)]],
                   uint t [[thread_index_in_threadgroup]]) {
    uint acc = 0u;
    uint n = t & 7u;
    for (uint i = 0u; i < n; i++) {
        acc += simd_sum(1u) * (i + 1u);
    }
    out[t] = acc;
}

// A loop with a per-thread continue and break.
kernel void break_k(device uint *out [[buffer(0)]],
                    uint t [[thread_index_in_threadgroup]]) {
    uint acc = 0u;
    for (uint i = 0u; i < 8u; i++) {
        if (i == (t & 3u)) {
            continue;
        }
        if (i > (t & 7u)) {
            break;
        }
        acc += simd_sum(i + 1u);
    }
    out[t] = acc;
}

// A return inside a loop.
kernel void loop_return_k(device uint *out [[buffer(0)]],
                          uint t [[thread_index_in_threadgroup]]) {
    uint acc = 0u;
    for (uint i = 0u; i < 4u; i++) {
        if (i == (t & 3u) && (t & 4u) != 0u) {
            out[t] = acc + 500u;
            return;
        }
        acc += simd_sum(1u);
    }
    out[t] = acc;
}

// A threadgroup barrier after an exit guard (Metal leaves a barrier that returned threads do not
// reach undefined; Apple GPUs release it with the threads that did not return, and the
// predicated emulation runs it in every thread).
kernel void guard_barrier_k(device uint *out [[buffer(0)]],
                            constant uint &limit [[buffer(1)]],
                            uint t [[thread_index_in_threadgroup]],
                            uint3 size [[threads_per_threadgroup]]) {
    threadgroup uint tg[128];
    if (t >= limit) {
        return;
    }
    tg[t] = t * 3u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint live = min(limit, size.x * size.y * size.z);
    out[t] = tg[(t + 1u) % live] + simd_sum(1u) * 1000u;
}

// A divergent switch.
kernel void switch_k(device uint *out [[buffer(0)]],
                     uint t [[thread_index_in_threadgroup]]) {
    uint r;
    switch (t & 3u) {
        case 0u:
            r = simd_sum(t);
            break;
        case 1u:
            r = simd_max(t) + 1000u;
            break;
        default:
            r = uint(simd_any(t == 6u)) + 7u;
            break;
    }
    out[t] = r;
}

// Quad operations after an exit guard.
kernel void quad_guard_k(device uint *out [[buffer(0)]],
                         uint t [[thread_index_in_threadgroup]]) {
    if ((t % 5u) == 3u) {
        return;
    }
    out[t] = quad_sum(t) + quad_shuffle_xor(t, ushort(1)) * 1000u
        + uint(quad_is_first()) * 1000000u
        + uint(quad_vote::vote_t(quad_active_threads_mask())) * 10000000u;
}

// A simdgroup barrier after an exit guard, the kernel's only simdgroup operation: each thread
// reads the word its simdgroup neighbour stored before the barrier.
kernel void guard_simd_barrier_k(device uint *out [[buffer(0)]],
                                 constant uint &limit [[buffer(1)]],
                                 uint t [[thread_index_in_threadgroup]],
                                 uint3 size [[threads_per_threadgroup]]) {
    threadgroup uint tg[128];
    if (t >= limit) {
        return;
    }
    tg[t] = t * 5u;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint live = min(limit, size.x * size.y * size.z);
    uint src = (t & ~31u) | ((t + 1u) & 31u);
    out[t] = src < live ? tg[src] : 7u;
}

// A simdgroup barrier, the kernel's only simdgroup operation, in a loop whose trip count is the
// thread's (the loop votes need the emulated tier's scratch with no exchange in the kernel).
kernel void loop_simd_barrier_k(device uint *out [[buffer(0)]],
                                uint t [[thread_index_in_threadgroup]]) {
    threadgroup uint tg[128];
    uint acc = 0u;
    for (uint i = 0u; i < (t & 3u); i++) {
        tg[t] = acc + i + 1u;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc = tg[t] * 2u;
    }
    out[t] = acc;
}

// A loop every thread runs together (its trip count and break are uniform) with a uniform
// `break` nested in a branch whose other paths rejoin the rest of the iteration, in a kernel
// predicated by a divergent operation after the loop: the break leaves the loop for every thread
// at once, skipping the rest of the iteration.
kernel void uniform_break_k(device uint *out [[buffer(0)]],
                            constant uint &limit [[buffer(1)]],
                            uint t [[thread_index_in_threadgroup]]) {
    uint n = limit & 7u;
    uint acc = 0u;
    for (uint i = 0u; i < 8u; i++) {
        acc += simd_sum(1u);
        if ((i & 1u) != 0u) {
            acc += simd_max(i);
            if (i == n) {
                break;
            }
        }
        acc += 1000u;
    }
    if ((t & 1u) != 0u) {
        acc += simd_sum(1u) * 100000u;
    }
    out[t] = acc;
}

// The same with a uniform `continue`: the rest of that iteration is skipped by every thread.
kernel void uniform_continue_k(device uint *out [[buffer(0)]],
                               constant uint &limit [[buffer(1)]],
                               uint t [[thread_index_in_threadgroup]]) {
    uint n = limit & 7u;
    uint acc = 0u;
    for (uint i = 0u; i < 8u; i++) {
        acc += simd_sum(1u);
        if ((i & 1u) != 0u) {
            acc += simd_max(i);
            if (i == n) {
                continue;
            }
        }
        acc += 1000u;
    }
    if ((t & 1u) != 0u) {
        acc += simd_sum(1u) * 100000u;
    }
    out[t] = acc;
}

// A uniform loop exit of its own (the loop's other exit leads elsewhere), holding a four-word
// exchange and a divergent branch: every thread takes the exit together, after the loop.
kernel void uniform_exit_k(device uint *out [[buffer(0)]],
                           constant uint &limit [[buffer(1)]],
                           uint t [[thread_index_in_threadgroup]]) {
    uint n = limit & 7u;
    uint acc = 0u;
    for (uint i = 0u; i < 8u; i++) {
        acc += simd_sum(1u);
        if (i == n) {
            uint4 v = simd_shuffle_xor(uint4(t, t + 1u, t + 2u, t + 3u), ushort(1));
            acc += v.x + v.w;
            if ((t & 1u) != 0u) {
                acc += simd_sum(1u) * 100000u;
            }
            out[t] = acc;
            return;
        }
        acc += 1000u;
    }
    out[t] = acc + 3u;
}
