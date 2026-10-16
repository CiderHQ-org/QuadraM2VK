// Owned fixture (NaN model, class J): which `float` operations read a denormal
// operand as a zero of its sign, and which flush a denormal result, in Metal's compile without
// fast math. Lane `i` reads the float patterns (a, b, c) from `in[i]` and writes 48 words to
// `out[12i .. 12i + 11]`: a + b, a - b, a * b, a / b, fma(a, b, c), a * b + c, -a, 0 - a,
// fabs(a), copysign(a, b), a < b and a == b (as 0 or 1); floor, ceil, trunc, round, rint, fract,
// sign, saturate, sqrt, rsqrt, log2, log, sin, tan, atan, asin, sinh, tanh, asinh, atanh of a;
// pow(a, b), powr(a, b), fmod(a, b), ldexp(a, 4), dot(float2(a, b), float2(b, c)),
// length(float2(a, b)), mix(a, b, c), step(a, b), smoothstep(a, b, c), atan2(a, b), fmin(a, b),
// (a * b) * c, (a / b) * c, a < b ? a : b, -(a * b), and fabs(a - b). `nan_denormals_fast` is
// the same program under the default fast math. Compiled without fast math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_denormals(device const uint4 *in [[buffer(0)]],
                          device uint4 *out [[buffer(1)]],
                          uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y), c = as_type<float>(in[tid].z);
    device uint4 *o = out + 12 * tid;
    o[0] = uint4(as_type<uint>(a + b), as_type<uint>(a - b), as_type<uint>(a * b), as_type<uint>(a / b));
    o[1] = uint4(as_type<uint>(fma(a, b, c)), as_type<uint>(a * b + c), as_type<uint>(-a),
                 as_type<uint>(0.0f - a));
    o[2] = uint4(as_type<uint>(fabs(a)), as_type<uint>(copysign(a, b)), uint(a < b), uint(a == b));
    o[3] = uint4(as_type<uint>(floor(a)), as_type<uint>(ceil(a)), as_type<uint>(trunc(a)),
                 as_type<uint>(round(a)));
    o[4] = uint4(as_type<uint>(rint(a)), as_type<uint>(fract(a)), as_type<uint>(sign(a)),
                 as_type<uint>(saturate(a)));
    o[5] = uint4(as_type<uint>(sqrt(a)), as_type<uint>(rsqrt(a)), as_type<uint>(log2(a)),
                 as_type<uint>(log(a)));
    o[6] = uint4(as_type<uint>(sin(a)), as_type<uint>(tan(a)), as_type<uint>(atan(a)),
                 as_type<uint>(asin(a)));
    o[7] = uint4(as_type<uint>(sinh(a)), as_type<uint>(tanh(a)), as_type<uint>(asinh(a)),
                 as_type<uint>(atanh(a)));
    o[8] = uint4(as_type<uint>(pow(a, b)), as_type<uint>(powr(a, b)), as_type<uint>(fmod(a, b)),
                 as_type<uint>(ldexp(a, 4)));
    o[9] = uint4(as_type<uint>(dot(float2(a, b), float2(b, c))), as_type<uint>(length(float2(a, b))),
                 as_type<uint>(mix(a, b, c)), as_type<uint>(step(a, b)));
    o[10] = uint4(as_type<uint>(smoothstep(a, b, c)), as_type<uint>(atan2(a, b)),
                  as_type<uint>(fmin(a, b)), as_type<uint>((a * b) * c));
    o[11] = uint4(as_type<uint>((a / b) * c), as_type<uint>(a < b ? a : b), as_type<uint>(-(a * b)),
                  as_type<uint>(fabs(a - b)));
}
