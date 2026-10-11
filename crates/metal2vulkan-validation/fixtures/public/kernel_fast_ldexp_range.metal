#include <metal_stdlib>
using namespace metal;

kernel void kernel_fast_ldexp_range(
    device float *out [[buffer(0)]],
    device const float *fin [[buffer(1)]],
    device const int *iin [[buffer(2)]])
{
    out[0] = ldexp(fin[0], iin[0]);
    out[1] = ldexp(fin[1], iin[1]);
    out[2] = ldexp(fin[2], iin[2]);
    out[3] = ldexp(fin[3], iin[3]);
    out[4] = ldexp(fin[4], iin[4]);
    out[5] = ldexp(fin[5], iin[5]);
    out[6] = ldexp(fin[6], iin[6]);
    out[7] = ldexp(fin[7], iin[7]);
}
