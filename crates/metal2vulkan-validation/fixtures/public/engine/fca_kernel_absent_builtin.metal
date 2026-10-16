// Owned fixture: a kernel builtin a function constant gates, read without a
// guard. MSL 5.8.1.2: with `has_width` false (or undefined) the argument is not declared in the
// function signature, and the language gives its read no value; the engine reads zero, the
// all-zero value of its type. With `has_width` true it is `threads_per_threadgroup`. Each thread
// stores what it read to `out[tid]`. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool has_width [[function_constant(0)]];

kernel void fca_kernel_absent_builtin(
    device uint *out [[buffer(0)]],
    uint width [[threads_per_threadgroup, function_constant(has_width)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = width;
}
