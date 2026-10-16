// Owned fixture (NaN model): `nan_minmax`' program under the default fast math, the
// flag state of class F that `nan_minmax` (without fast math) does not measure: `fmin`, `fmax`,
// `min3`, `max3`, `median3`, and `clamp` are the `fast_` calls, and `a < b ? a : b` and
// `a > b ? a : b` fold to `llvm.minnum` and `llvm.maxnum`, every call with `fast`. A NaN operand
// is undefined under fast math (MSL 7.1); the cases record what Metal answers. Lane `i` reads (a, b, c) from `in[i]` and writes eight words to `out[2i .. 2i + 1]`:
// fmin(a, b), fmax(a, b), a < b ? a : b, a > b ? a : b, min3, max3, median3, and clamp(a, b, c).
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void nan_minmax_fast(device const uint4 *in [[buffer(0)]],
                            device uint4 *out [[buffer(1)]],
                            uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y), c = as_type<float>(in[tid].z);
    out[2 * tid] = uint4(as_type<uint>(fmin(a, b)), as_type<uint>(fmax(a, b)),
                         as_type<uint>(a < b ? a : b), as_type<uint>(a > b ? a : b));
    out[2 * tid + 1] = uint4(as_type<uint>(min3(a, b, c)), as_type<uint>(max3(a, b, c)),
                             as_type<uint>(median3(a, b, c)), as_type<uint>(clamp(a, b, c)));
}
