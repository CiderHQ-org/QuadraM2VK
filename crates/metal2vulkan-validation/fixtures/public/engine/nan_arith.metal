// Owned fixture (NaN model): which NaN Metal's float arithmetic answers. Lane `i`
// reads the bit patterns (a, b, c) from `in[i]` and writes eight words to `out[2i .. 2i + 1]`:
// a + b, a - b, a * b, a / b, fma(a, b, c) (`air.fma`), a * b + c (`llvm.fmuladd`), sqrt(a)
// (`air.sqrt`), and fmod(a, b) (`air.fmod`). Every operand is loaded, so nothing folds.
// `nan_arith_fast` is the same program under the default fast math. Compiled without fast math,
// so a NaN, an infinity, or a signed zero is a defined value. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_arith(device const uint4 *in [[buffer(0)]],
                      device uint4 *out [[buffer(1)]],
                      uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y), c = as_type<float>(in[tid].z);
    out[2 * tid] = uint4(as_type<uint>(a + b), as_type<uint>(a - b), as_type<uint>(a * b),
                         as_type<uint>(a / b));
    out[2 * tid + 1] = uint4(as_type<uint>(fma(a, b, c)), as_type<uint>(a * b + c),
                             as_type<uint>(sqrt(a)), as_type<uint>(fmod(a, b)));
}
