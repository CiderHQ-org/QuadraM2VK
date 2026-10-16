// Owned fixture (NaN model): whether Metal's sign operations keep a NaN's bits. Lane
// `i` reads (a, b) from `in[i]` and writes eight words to `out[2i .. 2i + 1]`: fabs(a)
// (`air.fabs`; `abs` is the same call), -a (`fneg`), copysign(a, b) and copysign(b, a) (integer
// operations in the AIR), 0.0f - a (an `fsub`), -0.0f - a (folded to the `fneg`, as `a * -1.0f`
// is), sign(a) (`air.sign`), and saturate(a) (`air.saturate`). Compiled without fast math.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void nan_signbits(device const uint4 *in [[buffer(0)]],
                         device uint4 *out [[buffer(1)]],
                         uint tid [[thread_position_in_grid]])
{
    float a = as_type<float>(in[tid].x), b = as_type<float>(in[tid].y);
    out[2 * tid] = uint4(as_type<uint>(fabs(a)), as_type<uint>(-a), as_type<uint>(copysign(a, b)),
                         as_type<uint>(copysign(b, a)));
    out[2 * tid + 1] = uint4(as_type<uint>(0.0f - a), as_type<uint>(-0.0f - a),
                             as_type<uint>(sign(a)), as_type<uint>(saturate(a)));
}
