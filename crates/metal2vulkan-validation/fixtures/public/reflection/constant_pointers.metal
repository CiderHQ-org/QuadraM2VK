// Owned fixture: a program-scope pointer a module's static initializer
// chooses by a function constant, so it stores a pointer (a shape the crate's dependency analysis
// refused), read by one kernel and not by another. Metal's
// functionConstantsDictionary (the metallib's CNST tag) answers which constants each reads.
// Written for this repository; no third-party source.
#include <metal_stdlib>
using namespace metal;

constant bool cp_flag [[function_constant(0)]];
constant int cp_scale [[function_constant(1)]];

constant int cp_rows[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
constant int *constant cp_row = cp_flag ? cp_rows[0] : cp_rows[1];
constant int cp_scaled = cp_scale * 2;

kernel void cp_pointer(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = cp_row[tid & 3];
}

kernel void cp_value(device int *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = cp_scaled;
}
