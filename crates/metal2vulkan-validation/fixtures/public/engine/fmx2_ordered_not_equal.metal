// Owned fixture: an ordered not-equal and an unordered equal at NaN
// operands, without fast math (`-fno-fast-math`), so a NaN operand asks a defined question. The
// front end folds `a < b || a > b` into `fcmp one` (false when either operand is a NaN) and
// `!(a < b) && !(a > b)` into `fcmp ueq` (true when either is). SPIRV-Cross, which MoltenVK
// uses, renders `OpFOrdNotEqual` as MSL's `!=`, which is true for a NaN operand.
// metal-fast-math: off
//
// Thread i reads `in[i]` = float patterns (a, b) and writes four words, 1 or 0: the float
// `fcmp one` and `fcmp ueq` of (a, b), and the same of (half(a), half(b)). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void fmx2_ordered_not_equal(device const uint2 *in [[buffer(0)]],
                                   device uint4 *out [[buffer(1)]],
                                   uint i [[thread_position_in_grid]])
{
    float2 v = as_type<float2>(in[i]);
    float a = v.x, b = v.y;
    half ha = half(a), hb = half(b);
    out[i] = uint4((a < b || a > b) ? 1u : 0u, (!(a < b) && !(a > b)) ? 1u : 0u,
                   (ha < hb || ha > hb) ? 1u : 0u, (!(ha < hb) && !(ha > hb)) ? 1u : 0u);
}
