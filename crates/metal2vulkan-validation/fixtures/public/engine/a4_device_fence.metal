// Owned fixture: a device-scope memory fence between two stores
// (`barrier.fence`). out[t] = in[t] + 1 and out[t + 8] = 2 in[t], for eight threads. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_device_fence(device const uint *in [[buffer(0)]],
                            device uint *out [[buffer(1)]],
                            uint t [[thread_position_in_grid]]) {
    out[t] = in[t] + 1u;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    out[t + 8u] = in[t] * 2u;
}
