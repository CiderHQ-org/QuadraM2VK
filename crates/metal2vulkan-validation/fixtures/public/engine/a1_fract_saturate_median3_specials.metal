// Owned fixture: `fract`, `saturate`, and `median3` at NaN, infinities, signed
// zeros, and denormals, without fast math (`-fno-fast-math`: no call carries `nnan` or `ninf`,
// so every lane asks a defined question; `precise::` and `fast::` spellings are distinct `air`
// symbols with no flags). The translator lowers `fract` as `fmin(x - floor(x), 1 - ulp)`,
// `saturate` as a NaN-aware clamp, and `median3` through `NMin`/`NMax`; Metal's answers at
// these operands are unmeasured (the public kernel_a_nan_loses_to_a_number reads `saturate` of
// one NaN).
//
// Thread i reads `in[i]` = float patterns (x, a, b, c) and `hin[i]` = half patterns (x, a, b,
// c) and writes nine words: `precise::fract(x)`, `fast::fract(x)`, `precise::saturate(x)`,
// `fast::saturate(x)`, `precise::median3(a, b, c)`, `fast::median3(a, b, c)` (`air.fract.f32`,
// `air.fast_fract.f32`, `air.saturate.f32`, `air.fast_saturate.f32`, `air.fmedian3.f32`,
// `air.fast_fmedian3.f32`), then the half `fract(x) | saturate(x) << 16`, the half
// `median3(a, b, c)` (`air.fract.f16`, `air.saturate.f16`, `air.fmedian3.f16`) zero-extended,
// and 0. x runs over signed zeros, denormals, the smallest normals, values just below and above
// the integers (a negative x whose `x - floor(x)` rounds to 1), large integers and
// half-integers, the largest finite, infinities, and NaNs (quiet, signalling, both signs); the
// median triples place one, two, or three NaNs in every position, infinities and NaNs together,
// and zeros of both signs in every order. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void a1_fract_saturate_median3_specials(
    device const uint4 *in [[buffer(0)]],
    device const ushort4 *hin [[buffer(1)]],
    device uint *out [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    float4 v = as_type<float4>(in[i]);
    half4 h = as_type<half4>(hin[i]);
    device uint *o = out + 9 * i;
    o[0] = as_type<uint>(precise::fract(v.x));
    o[1] = as_type<uint>(fast::fract(v.x));
    o[2] = as_type<uint>(precise::saturate(v.x));
    o[3] = as_type<uint>(fast::saturate(v.x));
    o[4] = as_type<uint>(precise::median3(v.y, v.z, v.w));
    o[5] = as_type<uint>(fast::median3(v.y, v.z, v.w));
    o[6] = uint(as_type<ushort>(fract(h.x))) | uint(as_type<ushort>(saturate(h.x))) << 16;
    o[7] = uint(as_type<ushort>(median3(h.y, h.z, h.w)));
    o[8] = 0u;
}
