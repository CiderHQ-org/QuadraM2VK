// Owned fixture: the compiled replacement for the hand-written top-level
// `kernel_nested_loop_constant_weight_walk.ll`, whose SSA numbering `llvm-as` rejects. This is
// the same program as that fixture's `.metal` twin. A `constant` pointer walked by a nested loop:
// the inner loop advances it one `float4` per iteration, and because the inner trip count is the
// constant 3 the front end advances the outer loop's carried pointer by one `gep %outer, 3` on the
// outer backedge. Both loops stay loops, bounded by their constant trip counts (4 and 3), so
// authored cases are `authored_bounded`.
#include <metal_stdlib>
using namespace metal;

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
