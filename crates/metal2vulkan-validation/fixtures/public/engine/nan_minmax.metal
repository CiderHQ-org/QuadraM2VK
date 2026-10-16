// Owned fixture (NaN model): which NaN or number Metal's minimum and maximum answer.
// Lane `i` reads (a, b, c) from `in[i]` and writes eight words to `out[2i .. 2i + 1]`: fmin(a, b)
// and fmax(a, b) (`air.fmin`, `air.fmax`; `min` and `max` are the same calls), a < b ? a : b and
// a > b ? a : b (an `fcmp` and a `select`), min3, max3, median3, and clamp(a, b, c). Compiled
// without fast math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_minmax(device const uint4 *in [[buffer(0)]],
                       device uint4 *out [[buffer(1)]],
                       uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y), c = as_type<float>(in[tid].z);
    out[2 * tid] = uint4(as_type<uint>(fmin(a, b)), as_type<uint>(fmax(a, b)),
                         as_type<uint>(a < b ? a : b), as_type<uint>(a > b ? a : b));
    out[2 * tid + 1] = uint4(as_type<uint>(min3(a, b, c)), as_type<uint>(max3(a, b, c)),
                             as_type<uint>(median3(a, b, c)), as_type<uint>(clamp(a, b, c)));
}
