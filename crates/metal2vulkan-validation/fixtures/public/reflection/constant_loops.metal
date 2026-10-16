// Owned fixture: a module whose static initializer keeps loops, beside
// values computed from a loop's result and values computed without one. `constant_initializers`
// showed a loop's constants (its trip count and the value its body folds) in every function of
// the module, a select's only in the functions that read its global; this module asks which of a
// loop's neighbours spread too: a global computed from the loop's result and another constant, a
// global computed with no loop, and a loop of a literal trip count whose body reads a constant.
// Metal's functionConstantsDictionary (the metallib's CNST tag) answers. Written for this
// repository; no third-party source.
#include <metal_stdlib>
using namespace metal;

constant int cl_n [[function_constant(0)]];
constant int cl_a [[function_constant(1)]];
constant int cl_z [[function_constant(2)]];
constant int cl_w [[function_constant(3)]];
constant int cl_m [[function_constant(4)]];

// A loop of cl_n steps folding cl_a.
static int cl_steps(int n, int step) {
    int s = 1;
    for (int i = 0; i < n; i++) {
        s = s * step + i;
    }
    return s;
}
constant int cl_looped = cl_steps(cl_n, cl_a);

// The loop's result and another constant.
constant int cl_after = cl_looped + cl_z;

// No loop.
constant int cl_plain = cl_w * 3;

// A loop of a literal trip count whose body reads a constant.
static int cl_count(int m) {
    int s = 0;
    for (int i = 0; i < 1000; i++) {
        s = s * 7 + (i ^ m);
    }
    return s;
}
constant int cl_fixed = cl_count(cl_m);

kernel void cl_read_after(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = cl_after;
}

kernel void cl_read_plain(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = cl_plain;
}

kernel void cl_read_fixed(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = cl_fixed;
}

kernel void cl_none(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = int(tid);
}
