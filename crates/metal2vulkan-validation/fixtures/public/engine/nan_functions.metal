// Owned fixture (NaN model): which NaN Metal's float functions answer. Lane `i` reads
// the bit patterns (a, b, t) from `in[i]` (t a runtime blend factor) and writes 36 words to
// `out[9i .. 9i + 8]`: sin, cos, tan, asin, acos, atan of a; atan2(a, b); sinh, cosh, tanh, asinh,
// acosh, atanh, exp, exp2, exp10, log, log2, log10 of a; pow(a, b), powr(a, b); rsqrt, floor,
// ceil, round, trunc, rint, fract of a; dot(float2(a, b), float2(b, a)); length(float2(a, b));
// mix(a, b, t); step(a, b); smoothstep(b, t, a); ldexp(a, 2); and the patterns of a and b.
// `nan_functions_fast` is the same program under the default fast math. Compiled without fast
// math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_functions(device const uint4 *in [[buffer(0)]],
                          device uint4 *out [[buffer(1)]],
                          uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y), t = as_type<float>(in[tid].z);
    out[9 * tid + 0] = uint4(as_type<uint>(sin(a)), as_type<uint>(cos(a)), as_type<uint>(tan(a)),
                             as_type<uint>(asin(a)));
    out[9 * tid + 1] = uint4(as_type<uint>(acos(a)), as_type<uint>(atan(a)), as_type<uint>(atan2(a, b)),
                             as_type<uint>(sinh(a)));
    out[9 * tid + 2] = uint4(as_type<uint>(cosh(a)), as_type<uint>(tanh(a)), as_type<uint>(asinh(a)),
                             as_type<uint>(acosh(a)));
    out[9 * tid + 3] = uint4(as_type<uint>(atanh(a)), as_type<uint>(exp(a)), as_type<uint>(exp2(a)),
                             as_type<uint>(exp10(a)));
    out[9 * tid + 4] = uint4(as_type<uint>(log(a)), as_type<uint>(log2(a)), as_type<uint>(log10(a)),
                             as_type<uint>(pow(a, b)));
    out[9 * tid + 5] = uint4(as_type<uint>(powr(a, b)), as_type<uint>(rsqrt(a)), as_type<uint>(floor(a)),
                             as_type<uint>(ceil(a)));
    out[9 * tid + 6] = uint4(as_type<uint>(round(a)), as_type<uint>(trunc(a)), as_type<uint>(rint(a)),
                             as_type<uint>(fract(a)));
    out[9 * tid + 7] = uint4(as_type<uint>(dot(float2(a, b), float2(b, a))),
                             as_type<uint>(length(float2(a, b))), as_type<uint>(mix(a, b, t)),
                             as_type<uint>(step(a, b)));
    out[9 * tid + 8] = uint4(as_type<uint>(smoothstep(b, t, a)), as_type<uint>(ldexp(a, 2)),
                             in[tid].x, in[tid].y);
}
