// Owned fixture: program-scope constants a module's static initializer
// computes from function constants in the shapes the crate's dependency analysis refused: a
// definedness query a global stores, a loop whose trip count is a constant (a
// branch on it), and a choice between constants. Each is read by a kernel of its own, so Metal's
// functionConstantsDictionary (the metallib's CNST tag) answers which constants each kernel reads.
// Written for this repository; no third-party source.
#include <metal_stdlib>
using namespace metal;

constant int ci_a [[function_constant(0)]];
constant int ci_b [[function_constant(1)]];
constant int ci_c [[function_constant(2)]];
constant bool ci_flag [[function_constant(3)]];
constant int ci_n [[function_constant(4)]];
constant float ci_opt [[function_constant(5)]];
constant int ci_never [[function_constant(6)]];

// A definedness query a global stores.
constant bool ci_has_opt = is_function_constant_defined(ci_opt);

kernel void ci_defined_global(device float *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = ci_has_opt ? 2.0f : 1.0f;
}

// The stored query and the value it guards.
kernel void ci_defined_and_read(device float *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = ci_has_opt ? ci_opt : 1.0f;
}

// A branch on a constant: a loop of ci_n steps, a recurrence no closed form replaces.
static int ci_steps(int n, int step) {
    int s = 1;
    for (int i = 0; i < n; i++) {
        s = s * step + i;
    }
    return s;
}
constant int ci_looped = ci_steps(ci_n, ci_a);

kernel void ci_branch(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = ci_looped;
}

// A choice between two constants by a third.
constant int ci_chosen = ci_flag ? ci_b : ci_c;

kernel void ci_choice(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = ci_chosen;
}

// No constant at all.
kernel void ci_none(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = int(tid);
}
