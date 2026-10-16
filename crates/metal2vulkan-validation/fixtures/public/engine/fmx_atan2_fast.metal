// Owned fixture: `fmx_atan2_precise` compiled with the default fast math.
// `atan2` and `fast::atan2` are `air.fast_atan2.f32` (one call), `precise::atan2` is
// `air.atan2.f32`, and the half call is `air.atan2.f16`, each carrying LLVM's fast flags, under
// which the special operands are undefined (MSL section 7.1). The case reads which answers Metal
// gives there all the same.
//
// Thread i reads `fin[i]` = float patterns (y, x) and `hin[i]` = half patterns (y, x), and writes
// four words: `atan2(y, x)`, `fast::atan2(y, x)`, the half `atan2(y, x)` in the low half, and
// `precise::atan2(y, x)`. The operands are `fmx_atan2_precise`'s. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void fmx_atan2_fast(device const uint2 *fin [[buffer(0)]],
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
