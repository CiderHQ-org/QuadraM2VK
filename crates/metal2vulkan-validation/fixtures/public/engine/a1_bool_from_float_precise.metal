// Owned fixture: `air.convert` from a float to `i1`, without fast math. The same
// program as `a1_bool_from_float`, compiled `-fno-fast-math` (the module records
// `air.compile.fast_math_disable`), so a disagreement between the two on a NaN, an infinity, or a
// denormal lane shows the conversion depends on the module's fast-math option. Loop-free.
// metal-fast-math: off
//
// Thread i converts four `float` patterns (`bits[i]`) and four `half` patterns (`half_bits[i]`),
// each lane once as a scalar and once in a vector, and writes one word: bits 0-3 the scalar
// `float` conversions, 4-7 the vector `float` ones, 8-11 the scalar `half` ones, 12-15 the vector
// `half` ones (lane x lowest); bits 16-31 are 0.
#include <metal_stdlib>
using namespace metal;

kernel void a1_bool_from_float_precise(
    device const uint4 *bits [[buffer(0)]],
    device const ushort4 *half_bits [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    float4 x = as_type<float4>(bits[i]);
    half4 h = as_type<half4>(half_bits[i]);
    bool4 xv = bool4(x);
    bool4 hv = bool4(h);
    uint r = uint(bool(x.x)) | uint(bool(x.y)) << 1 | uint(bool(x.z)) << 2 | uint(bool(x.w)) << 3;
    r |= uint(xv.x) << 4 | uint(xv.y) << 5 | uint(xv.z) << 6 | uint(xv.w) << 7;
    r |= uint(bool(h.x)) << 8 | uint(bool(h.y)) << 9 | uint(bool(h.z)) << 10 | uint(bool(h.w)) << 11;
    r |= uint(hv.x) << 12 | uint(hv.y) << 13 | uint(hv.z) << 14 | uint(hv.w) << 15;
    out[i] = r;
}
