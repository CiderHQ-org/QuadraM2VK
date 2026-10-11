#include <metal_stdlib>
using namespace metal;

// Semantic twin of the hand-written AIR beside it. Both compare DEVICE POINTERS: `p` against a
// second pointer whose index is a different expression, and `p` against a fixed element. The
// translator has no addresses to compare -- Logical SPIR-V pointers are not values -- so it has to
// answer both from the access-chain index paths, which is the shape under test.
kernel void compare_two_element_pointers(
    device const float* buf [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    device const float* p = &buf[gid];
    device const float* q = &buf[(gid >> 1) << 1];
    device const float* r = &buf[3];
    out[gid]     = ((p == q) ? 100.0f : 0.0f) + *p;
    out[gid + 8] = ((p != r) ? 1000.0f : 0.0f) + *r;
}
