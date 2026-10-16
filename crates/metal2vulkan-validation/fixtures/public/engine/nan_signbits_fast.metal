// Owned fixture (NaN model): `nan_signbits`' program under the default fast math,
// where `fabs` is `air.fast_fabs`, `-a`, `0.0f - a`, and `-0.0f - a` are one `fneg fast`, and
// `saturate` is `air.fast_saturate`. A NaN or infinite operand is undefined under fast math (MSL
// 7.1); the case records what Metal answers, the flag state of class E that `nan_signbits`
// (without fast math) does not measure. Lane `i` reads (a, b) from `in[i]` and
// writes eight words to `out[2i .. 2i + 1]`: fabs(a), -a, copysign(a, b), copysign(b, a),
// 0.0f - a, -0.0f - a, sign(a), and saturate(a). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void nan_signbits_fast(device const uint4 *in [[buffer(0)]],
                              device uint4 *out [[buffer(1)]],
                              uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y);
    out[2 * tid] = uint4(as_type<uint>(fabs(a)), as_type<uint>(-a), as_type<uint>(copysign(a, b)),
                         as_type<uint>(copysign(b, a)));
    out[2 * tid + 1] = uint4(as_type<uint>(0.0f - a), as_type<uint>(-0.0f - a),
                             as_type<uint>(sign(a)), as_type<uint>(saturate(a)));
}
