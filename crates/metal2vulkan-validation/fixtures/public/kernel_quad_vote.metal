#include <metal_stdlib>
using namespace metal;

kernel void kernel_quad_vote(device const uint *in [[buffer(0)]],
                             device uint *out [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
    bool p = in[tid] != 0u;
    out[tid * 2u + 0u] = quad_all(p) ? 1u : 0u;
    out[tid * 2u + 1u] = quad_any(p) ? 1u : 0u;
}
