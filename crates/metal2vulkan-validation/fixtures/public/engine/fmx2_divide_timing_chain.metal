// Owned fixture: a timing kernel for float division without fast math
// (`-fno-fast-math`) whose divisor is the running value, so neither the division nor any part
// of it is loop-invariant (fmx2_divide_timing's divisor is, and a device compiler may hoist its
// reciprocal). Each thread runs, each step waiting for the last:
//
//   acc = c / acc + 0.5   (n times)
//
// which settles near the root of acc^2 - acc / 2 - c. Thread i reads n = in[0], its numerator's
// bits in[1 + i % 64], and its start value's bits in[65 + i % 64], and writes the final value's
// bits to out[i]. Evidence scripts generate the cases (n, the numerators, the thread count) and
// may add `arcp` to the `fdiv` to time the fast division; the AIR's one loop is bounded by n.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void fmx2_divide_timing_chain(device const uint *in [[buffer(0)]],
                                     device uint *out [[buffer(1)]],
                                     uint i [[thread_position_in_grid]])
{
    uint n = in[0];
    float c = as_type<float>(in[1 + (i & 63)]);
    float acc = as_type<float>(in[65 + (i & 63)]);
    for (uint k = 0; k < n; ++k) {
        acc = c / acc + 0.5f;
    }
    out[i] = as_type<uint>(acc);
}
