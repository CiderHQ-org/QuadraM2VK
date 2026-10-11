#include <metal_stdlib>
using namespace metal;

// The sibling .ll is the SAME automaton written as a three-block cycle that %entry's switch enters
// at ALL THREE of its blocks, which is irreducible and which the Metal front end cannot emit from
// any source text. This file is the reducible spelling: one loop, one `state` variable, the same
// three arithmetic steps, and the same slot choice at the exit.
kernel void irreducible_three_way(device uint *output [[buffer(0)]],
                                  const device uint *input [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]]) {
    uint n = input[tid];
    uint state = n % 3u;
    uint v = state == 0u ? 10u : (state == 1u ? 20u : 30u);
    uint i = 0u;
    for (;;) {
        if (state == 0u) {
            v = v * 3u;
        } else if (state == 1u) {
            v = v + 7u;
        } else {
            v = v ^ 255u;
        }
        i = i + 1u;
        if (i >= n) {
            break;
        }
        state = (state + 1u) % 3u;
    }
    output[tid * 4u + state] = v;
}
