#include <metal_stdlib>
using namespace metal;

kernel void kernel_large_vector_dynamic_lane(device uint *out [[buffer(0)]],
                                             uint tid [[thread_position_in_grid]]) {
    thread uint v[8] = {1u, 11u, 21u, 31u, 41u, 51u, 61u, 71u};
    uint ins = tid & 7u;
    v[ins] = tid + 1000u;
    uint ext = (tid * 5u) & 7u;
    out[tid * 2u + 0u] = v[ext];
    out[tid * 2u + 1u] = v[ins];
}
