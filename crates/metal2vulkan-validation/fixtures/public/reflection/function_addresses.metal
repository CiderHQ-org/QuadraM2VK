// Owned fixture: a kernel that takes the addresses of visible functions
// which read function constants and calls one through them, beside a kernel that calls one
// directly and one that calls a plain function out of line. Metal's functionConstantsDictionary
// (the metallib's CNST tag) answers whether a kernel reads the constants of the functions whose
// addresses it takes or that it calls. Written for this repository; no third-party source.
#include <metal_stdlib>
using namespace metal;

constant int fa_add_k [[function_constant(0)]];
constant int fa_sub_k [[function_constant(1)]];
constant int fa_own [[function_constant(2)]];
constant int fa_help_k [[function_constant(3)]];

[[visible]] int fa_add(int x) { return x + fa_add_k; }
[[visible]] int fa_sub(int x) { return x - fa_sub_k; }

kernel void fa_pointers(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    int (*fs[2])(int) = {fa_add, fa_sub};
    out[tid] = fs[tid & 1](int(tid)) + fa_own;
}

kernel void fa_direct(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = fa_add(int(tid));
}

// A plain function the compiler keeps out of line, called directly: its constant against the
// visible function's above.
__attribute__((noinline)) int fa_help(int x) { return x * fa_help_k; }

kernel void fa_plain(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = fa_help(int(tid));
}
