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
