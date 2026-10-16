// Owned fixture: Metal's precise `float` division at the edges a correctly
// rounded quotient has to get right, compiled without fast math (`-fno-fast-math`), so `x / y` is
// an `fdiv float` with no flag. A Metal probe found the flagless quotient correctly rounded on
// every lane, with an `f32` denormal operand read as a zero and a denormal quotient flushed; the
// lanes here separate where the flush is decided (the quotient rounded to the denormal grid, or
// to 24 bits first) and test quotients within a hair of a rounding midpoint.
//
//   word   operation
//   0      x / y
//
// Thread i reads `in[4 i + j]`: x, y at j = 0, 1 (j = 2, 3 unused), and writes `out[4 i]`; words
// 1-3 keep the output's initial bytes. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void fmx2_divide_edges(device const float *in [[buffer(0)]],
                              device uint *out [[buffer(1)]],
                              uint i [[thread_position_in_grid]])
{
    device const float *v = in + 4 * i;
    out[4 * i] = as_type<uint>(v[0] / v[1]);
}
