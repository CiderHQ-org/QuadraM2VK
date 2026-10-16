// Owned fixture (NaN model): `nan_arith`'s program under the default fast math, where
// the operations carry `fast` (`fadd`, `fsub`, `fmul`, `fdiv`, `air.fma`), `a * b + c` is an
// `fmul fast` feeding an `fadd fast`, and sqrt and fmod are `air.fast_sqrt` and `air.fast_fmod`.
// A NaN or infinite operand is undefined under fast math (MSL 7.1); the cases record what Metal
// answers. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void nan_arith_fast(device const uint4 *in [[buffer(0)]],
                           device uint4 *out [[buffer(1)]],
                           uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y), c = as_type<float>(in[tid].z);
    out[2 * tid] = uint4(as_type<uint>(a + b), as_type<uint>(a - b), as_type<uint>(a * b),
                         as_type<uint>(a / b));
    out[2 * tid + 1] = uint4(as_type<uint>(fma(a, b, c)), as_type<uint>(a * b + c),
                             as_type<uint>(sqrt(a)), as_type<uint>(fmod(a, b)));
}
