#include <metal_stdlib>
using namespace metal;

// A `constant` pointer walked by a nested loop: the inner loop advances it and the
// outer loop carries the advanced pointer round again, so the induction pointer is a
// cycle of phis rather than a single self-recursive one.
kernel void nested_loop_constant_weight_walk(constant float4 *weights [[buffer(0)]],
                                             device float *out [[buffer(1)]],
                                             uint gid [[thread_position_in_grid]]) {
    constant float4 *p = weights;
    float acc = float(gid);
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 3u; ++j) {
            acc += p[0].x;
            p += 1;
        }
    }
    out[gid] = acc;
}
