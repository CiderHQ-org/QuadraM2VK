// metal-std: metal4.0
// metal-macos-min: 27.0
// Owned fixture: the AIR 2.9 twin of
// validation/fixtures/public/engine/a4_atomic_load_only.metal. Below the next blank line is
// that file, unchanged; the directives above compile it as MSL 4.0 for macOS 27, where Apple's
// compiler writes AIR 2.9 (each memory atomic with its `mem_flags`, each barrier with its
// `memory_order`). It compiles on every committed profile to the same words as its original
// (tests/engine_air_2_9.rs), so the original's cases and rows speak for it.

// Owned fixture: a `device` atomic that is only loaded
// (`access.atomic.buffer.load`). Every thread reads the limit n with a relaxed atomic load:
// out[i] = i < n ? a[i] : 1000 + n. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_atomic_load_only(device atomic_uint &limit [[buffer(0)]],
                                device const uint *a [[buffer(1)]],
                                device uint *out [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    uint n = atomic_load_explicit(&limit, memory_order_relaxed);
    out[i] = i < n ? a[i] : 1000u + n;
}
