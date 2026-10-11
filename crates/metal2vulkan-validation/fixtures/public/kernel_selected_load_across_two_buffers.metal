#include <metal_stdlib>
using namespace metal;

// Two DISTINCT device buffers reached through one selected pointer. Logical SPIR-V cannot select
// pointers whose roots are different descriptor variables, so interface substitution has to
// materialize the loads per arm and select the VALUE instead.
kernel void kernel_selected_load_across_two_buffers(device const uint *a [[buffer(0)]],
                                                    device const uint *b [[buffer(1)]],
                                                    device const uint *pick [[buffer(2)]],
                                                    device uint *out [[buffer(3)]],
                                                    uint tid [[thread_position_in_grid]]) {
    bool take_a = pick[tid] != 0u;
    // One arm is an offset into `a`, the other is `b` itself. Both arms being offsets does NOT
    // reach the rule -- the bare descriptor root on one side is what keeps the select alive.
    device const uint *p = take_a ? (a + tid) : b;
    device const uint *q = take_a ? (a + tid + 1u) : b;
    out[tid] = *p + *q * 3u;
}
