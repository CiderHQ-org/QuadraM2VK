// Owned fixture: how Metal reads `float` denormals in vector
// arithmetic, comparisons and functions, and in operations whose operand is another operation's
// result, in Metal's compile without fast math. Thread `i` reads the float vectors (a, b, c) and
// the int vector n from `in[4i .. 4i + 3]` and writes 64 words to `out[16i .. 16i + 15]`, each row
// lane-wise: a + b, a - b, a * b, fma(a, b, c), a * b + c, the six comparisons of a and b as bits
// (<, <=, ==, !=, >, >= at bits 0 to 5), floor(a), ceil(a), sqrt(a), rsqrt(a), ldexp(a, n),
// (a * b) * c, float(n) * a - c, the comparisons a * b < c, -a < b and fabs(a) == b as bits 0 to 2,
// fmin(a, b) * c, and step(a, b). `dfl_float_denormals_fast` is the same program under the default
// fast math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void dfl_float_denormals(device const uint4 *in [[buffer(0)]],
                                device uint4 *out [[buffer(1)]],
                                uint tid [[thread_position_in_grid]])
{
    float4 a = as_type<float4>(in[4 * tid]);
    float4 b = as_type<float4>(in[4 * tid + 1]);
    float4 c = as_type<float4>(in[4 * tid + 2]);
    int4 n = as_type<int4>(in[4 * tid + 3]);
    device uint4 *o = out + 16 * tid;
    o[0] = as_type<uint4>(a + b);
    o[1] = as_type<uint4>(a - b);
    o[2] = as_type<uint4>(a * b);
    o[3] = as_type<uint4>(fma(a, b, c));
    o[4] = as_type<uint4>(a * b + c);
    o[5] = uint4(a < b) | uint4(a <= b) << 1 | uint4(a == b) << 2 | uint4(a != b) << 3 |
           uint4(a > b) << 4 | uint4(a >= b) << 5;
    o[6] = as_type<uint4>(floor(a));
    o[7] = as_type<uint4>(ceil(a));
    o[8] = as_type<uint4>(sqrt(a));
    o[9] = as_type<uint4>(rsqrt(a));
    o[10] = as_type<uint4>(ldexp(a, n));
    o[11] = as_type<uint4>((a * b) * c);
    o[12] = as_type<uint4>(float4(n) * a - c);
    o[13] = uint4(a * b < c) | uint4(-a < b) << 1 | uint4(fabs(a) == b) << 2;
    o[14] = as_type<uint4>(fmin(a, b) * c);
    o[15] = as_type<uint4>(step(a, b));
}
