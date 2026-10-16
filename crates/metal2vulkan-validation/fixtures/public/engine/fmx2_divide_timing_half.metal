// Owned fixture: a timing kernel for half division, compiled without fast math (`-fno-fast-math`),
// so `acc / y` is an `fdiv half` with no flag, a division the translator makes Metal's half
// quotient (rounded once from the exact one) on every profile. Each thread divides its running
// value by its divisor n times, each step waiting for the last:
//
//   acc = acc / y + 0.5h   (n times)
//
// Thread i reads n = in[0], its divisor's bits in the low half of in[1 + i % 64], and its start
// value's bits in the low half of in[65 + i % 64], and writes the final value's bits to out[i].
// Evidence scripts generate the cases (n, the divisors, the thread count); the AIR's one loop is
// bounded by n.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void fmx2_divide_timing_half(device const uint *in [[buffer(0)]],
                                    device uint *out [[buffer(1)]],
                                    uint i [[thread_position_in_grid]])
{
    uint n = in[0];
    half y = as_type<half>(ushort(in[1 + (i & 63)]));
    half acc = as_type<half>(ushort(in[65 + (i & 63)]));
    for (uint k = 0; k < n; ++k) {
        acc = acc / y + 0.5h;
    }
    out[i] = uint(as_type<ushort>(acc));
}
