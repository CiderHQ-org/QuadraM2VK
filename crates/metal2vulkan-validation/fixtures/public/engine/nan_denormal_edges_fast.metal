// Owned fixture (NaN model, class J): `nan_denormal_edges`' program under the default
// fast math, its flagged state: whether Metal flushes a `float` result that is below 2^-126
// before rounding but rounds to 2^-126. Lane `i` reads (a, b, c, n) from `in[i]` and writes 8
// words to `out[2i .. 2i + 1]`, `nan_denormal_edges`' fields. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void nan_denormal_edges_fast(device const uint4 *in [[buffer(0)]],
                                    device uint4 *out [[buffer(1)]],
                                    uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y), c = as_type<float>(in[tid].z);
    int n = as_type<int>(in[tid].w);
    device uint4 *o = out + 2 * tid;
    o[0] = uint4(as_type<uint>(a * b), as_type<uint>(a / b), as_type<uint>(fma(a, b, c)),
                 as_type<uint>(a * b + c));
    o[1] = uint4(as_type<uint>(ldexp(a, n)), as_type<uint>(-(a * b)), as_type<uint>(-(a / b)),
                 as_type<uint>(fma(-a, b, -c)));
}
