// Owned fixture: Metal's `atan2` at every pair of IEEE special operands,
// compiled without fast math (`-fno-fast-math`), so no call carries `nnan`, `ninf`, or `nsz` and
// C99 Annex F's answers are defined. `atan2` and `precise::atan2` are `air.atan2.f32` (one call),
// `fast::atan2` is `air.fast_atan2.f32`, and the half call is `air.atan2.f16`.
// `fmx_atan2_fast` is the same program with the default fast math.
// metal-fast-math: off
//
// Thread i reads `fin[i]` = float patterns (y, x) and `hin[i]` = half patterns (y, x), and writes
// four words: `atan2(y, x)`, `fast::atan2(y, x)`, the half `atan2(y, x)` in the low half, and
// `precise::atan2(y, x)`. The operands are every pair of sixteen values per width: both zeros,
// both smallest denormals, the smallest normal, 1, -1, 3, -0.75, both largest finites, both
// infinities, and three NaNs (positive and negative quiet, and one with a payload). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void fmx_atan2_precise(device const uint2 *fin [[buffer(0)]],
                              device const ushort2 *hin [[buffer(1)]],
                              device uint4 *out [[buffer(2)]],
                              uint i [[thread_position_in_grid]])
{
    float2 f = as_type<float2>(fin[i]);
    half2 h = as_type<half2>(hin[i]);
    out[i] = uint4(as_type<uint>(atan2(f.x, f.y)), as_type<uint>(fast::atan2(f.x, f.y)),
                   uint(as_type<ushort>(atan2(h.x, h.y))),
                   as_type<uint>(precise::atan2(f.x, f.y)));
}
