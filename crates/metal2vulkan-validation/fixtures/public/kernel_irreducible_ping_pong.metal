#include <metal_stdlib>
using namespace metal;

// The sibling .ll is the SAME automaton written with two entries into its cycle, which is
// irreducible and which the Metal front end cannot emit from any source text. This file is the
// reducible spelling of it: one loop, one `in_b` state bit, the same two arithmetic steps.
kernel void irreducible_ping_pong(device uint *output [[buffer(0)]],
                                  const device uint *input [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]]) {
    uint n = input[tid];
    bool in_b = (n & 1u) == 1u;
    uint acc = in_b ? 2u : 1u;
    uint i = 0u;
    for (;;) {
        acc = in_b ? (acc + 5u) : (acc * 3u);
        i = i + 1u;
        if (i >= n) {
            break;
        }
        in_b = !in_b;
    }
    output[tid] = acc;
}
