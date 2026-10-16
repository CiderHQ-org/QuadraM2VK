// Owned fixture: a module whose static initializer branches on a
// function constant without a loop, beside a loop in a function it calls out of line.
// `constant_initializers` showed a loop's constants (its trip count and the value its body folds)
// in every function of the module, a select's only in the functions that read its global; this
// module asks which constants of an if/else spread: one whose arm divides (a division the
// compiler cannot run on the other arm, so a branch on `cb_flag` and a merge of the two arms'
// values), and a loop that only an out-of-line callee of the initializer runs. Metal's
// functionConstantsDictionary (the metallib's CNST tag) answers. Written for this repository; no
// third-party source.
#include <metal_stdlib>
using namespace metal;

constant bool cb_flag [[function_constant(0)]];
constant int cb_p [[function_constant(1)]];
constant int cb_d [[function_constant(2)]];
constant int cb_q [[function_constant(3)]];
constant int cb_k [[function_constant(4)]];
constant int cb_s [[function_constant(5)]];

// An if/else: the division runs on one arm only.
constant int cb_div = cb_flag ? cb_p / cb_d : cb_q;

// A loop of cb_k steps folding cb_s, in a function the initializer calls out of line.
__attribute__((noinline)) static int cb_loop(int n, int step) {
    int s = 1;
    for (int i = 0; i < n; i++) {
        s = s * step + i;
    }
    return s;
}
constant int cb_called = cb_loop(cb_k, cb_s);

kernel void cb_read_div(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = cb_div;
}

kernel void cb_read_called(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = cb_called;
}

kernel void cb_none(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = int(tid);
}
