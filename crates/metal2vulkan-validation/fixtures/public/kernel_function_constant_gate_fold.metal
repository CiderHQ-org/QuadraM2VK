#include <metal_stdlib>
using namespace metal;

// `mode` is a function constant supplied as 0. The translator models every function
// constant at its disabled default, so every arm this gates is statically dead and the
// folds under test -- select, then branch, then unreachable-block removal -- all fire.
constant int mode [[function_constant(0)]];

kernel void fcfold(device const float *in [[buffer(0)]],
                   device float *out [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    float v = in[gid];

    // Ungated.
    out[gid] = v + 1.0f;

    // A gated BRANCH: the whole block is unreachable at mode == 0.
    if (mode != 0) {
        out[gid + 8] = v * 1000.0f;
    }

    // A gated SELECT feeding an ungated write.
    out[gid + 16] = (mode > 0) ? (v * 1000.0f) : (v - 1.0f);

    // A gated value merged from several arms, all of which are the same at mode == 0.
    float acc = v;
    if (mode == 1)       { acc = v * 1000.0f; }
    else if (mode == 2)  { acc = v * 2000.0f; }
    else if (mode == 3)  { acc = v * 3000.0f; }
    out[gid + 24] = acc;
}
