// Owned fixture: whether Metal's half-to-float widening of a NaN answers
// the canonical NaN outside any texture path. Lane `i` loads the raw 16-bit pattern `in[i]` from a
// buffer, reinterprets it as `half`, widens it to `float`, and stores to `out[i]` the widened bits,
// the bits of the widening times the literal 1.0f, the bits of the widening times `one[0]` (a
// runtime 1.0f the compiler cannot fold), and the raw pattern. Compiled without fast math, so a NaN
// is a defined value. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void hrd_half_widen(device const ushort *in [[buffer(0)]],
                           device const float *one [[buffer(1)]],
                           device uint4 *out [[buffer(2)]],
                           uint tid [[thread_position_in_grid]])
{
    half h = as_type<half>(in[tid]);
    float w = float(h);
    out[tid] = uint4(as_type<uint>(w), as_type<uint>(w * 1.0f), as_type<uint>(w * one[0]),
                     uint(in[tid]));
}
