// metal-std: metal4.0
// metal-macos-min: 27.0
// Owned fixture: the AIR 2.9 twin of
// validation/fixtures/public/engine/ob1_buffer_atomic.metal. Below the next blank line is that
// file, unchanged; the directives above compile it as MSL 4.0 for macOS 27, where Apple's compiler
// writes AIR 2.9 (each memory atomic with its `mem_flags`, each barrier with its `memory_order`).
// It compiles on every committed profile to the same words as its original
// (tests/engine_air_2_9.rs), so the original's cases and rows speak for it.

// Owned fixture: atomics past a `device` buffer, the buffer atomic
// containment form. Eight threads; thread t adds 1 + t to counter idx[t] of ctr and stores the
// value the add returned to ctr[4 + t]. The case's idx puts adds 0 to 3 on counters 0 to 3 (one
// thread each) and adds 4 to 7 past ctr's end (words 12, 13, 40, 75 of a 12-word ctr: up to byte
// 304). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void ob1_buffer_atomic(device const uint *idx [[buffer(0)]],
                              device atomic_uint *ctr [[buffer(1)]],
                              uint t [[thread_position_in_grid]]) {
    uint old = atomic_fetch_add_explicit(&ctr[idx[t]], 1u + t, memory_order_relaxed);
    atomic_store_explicit(&ctr[4 + t], old, memory_order_relaxed);
}
