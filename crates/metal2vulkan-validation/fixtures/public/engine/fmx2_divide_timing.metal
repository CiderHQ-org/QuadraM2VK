// Owned fixture: a timing kernel for float division without fast math (`-fno-fast-math`), so `acc /
// y` is an `fdiv float` with no flag, the division the translator makes Metal's correctly rounded
// quotient. Each thread divides its running value by its divisor n times, each step waiting for the
// last:
//
//   acc = acc / y + 0.5   (n times)
//
// Thread i reads n = in[0], its divisor's bits in[1 + i % 64], and its start value's bits
// in[65 + i % 64], and writes the final value's bits to out[i]. Evidence scripts generate the
// cases (n, the divisors, the thread count); the AIR's one loop is bounded by n.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void fmx2_divide_timing(device const uint *in [[buffer(0)]],
                               device uint *out [[buffer(1)]],
                               uint i [[thread_position_in_grid]])
{
    uint n = in[0];
    float y = as_type<float>(in[1 + (i & 63)]);
    float acc = as_type<float>(in[65 + (i & 63)]);
    for (uint k = 0; k < n; ++k) {
        acc = acc / y + 0.5f;
    }
    out[i] = as_type<uint>(acc);
}
