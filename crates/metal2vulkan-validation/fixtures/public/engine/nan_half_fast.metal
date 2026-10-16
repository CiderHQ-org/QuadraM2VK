// Owned fixture (NaN model): `nan_half`'s program under the default fast math, where
// the half arithmetic and functions carry `fast` (a `half` function keeps the precise spelling)
// and the widenings carry no flags. A NaN or infinite operand is undefined under fast math (MSL
// 7.1); the case records what Metal answers, the flagged state of the half classes. Lane `i`
// reads the half patterns a, b, t, and c and writes `nan_half`'s 32 words. Loop-free.
#include <metal_stdlib>
using namespace metal;

static uint pack(half lo, half hi)
{
    return uint(as_type<ushort>(lo)) | uint(as_type<ushort>(hi)) << 16;
}

kernel void nan_half_fast(device const uint4 *in [[buffer(0)]],
                          device uint4 *out [[buffer(1)]],
                          uint tid [[thread_position_in_grid]])
{
    half a = as_type<half>(ushort(in[tid].x)), b = as_type<half>(ushort(in[tid].x >> 16));
    half t = as_type<half>(ushort(in[tid].y)), c = as_type<half>(ushort(in[tid].y >> 16));
    device uint4 *o = out + 8 * tid;
    o[0] = uint4(as_type<uint>(float(a)), as_type<uint>(float(b)), pack(a - b, a / b),
                 pack(a * b + c, sin(a)));
    o[1] = uint4(pack(cos(a), tan(a)), pack(asin(a), acos(a)), pack(atan(a), atan2(a, b)),
                 pack(sinh(a), cosh(a)));
    o[2] = uint4(pack(tanh(a), asinh(a)), pack(acosh(a), atanh(a)), pack(exp(a), exp2(a)),
                 pack(exp10(a), log(a)));
    o[3] = uint4(pack(log2(a), log10(a)), pack(pow(a, b), powr(a, b)), pack(sqrt(a), rsqrt(a)),
                 pack(floor(a), ceil(a)));
    o[4] = uint4(pack(round(a), trunc(a)), pack(rint(a), fract(a)),
                 pack(dot(half2(a, b), half2(b, a)), length(half2(a, b))),
                 pack(mix(a, b, t), step(a, b)));
    o[5] = uint4(pack(smoothstep(b, t, a), ldexp(a, 2)), pack(fmin(a, b), fmax(a, b)),
                 pack(fabs(a), min3(a, b, t)), pack(max3(a, b, t), median3(a, b, t)));
    o[6] = uint4(pack(clamp(a, b, t), sign(a)), pack(saturate(a), copysign(a, b)),
                 pack(fmod(a, b), -a), pack(a * b, a + b));
    o[7] = uint4(as_type<uint>(float(t)), as_type<uint>(float(c)), in[tid].x, in[tid].y);
}
