// Owned fixture: what Metal's `half` `tanh` computes under fast math. MSL has no half fast `tanh`:
// `tanh(half)` in a fast-math module is `air.tanh.f16` (or `air.tanh.v4f16`) with the `fast` flags,
// the form of every half `tanh` in the corpus, while `fast::tanh` of a half promotes it to `float`
// (`air.fast_tanh.f32`). The translator lowers `air.fast_tanh.f32` as the fast-math formula `(t -
// 1) / (t + 1)`, `t = exp(2x)` (which answers 0 near x = 44 and NaN from 44.4 on), and keeps `Tanh`
// for the half symbols; whether Metal's half `tanh` follows the `f32` formula, a half-precision
// formula (whose `exp(2x)` overflows past x = 5.55), or `tanh` itself is unmeasured (the narrow
// forms are lowered through the `f32` formula only if this reading agrees). Compiled with the
// default fast math, as the corpus is.
//
// Thread t (0..4095) takes the half h of pattern 16t (every sixteenth pattern: both signs, every
// binade from the denormals to 65504 at 64 points, the infinities, and NaNs) and writes two words:
// `fast::tanh(float(h))` (`air.fast_tanh.f32`, the `f32` formula on the same input), then
// `tanh(h)` (`air.tanh.f16`) in the low half and lane x of `tanh(half4(h, -h, h, -h))`
// (`air.tanh.v4f16`) in the high half. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a1_half_tanh_regimes(
    device uint *out [[buffer(0)]],
    uint t [[thread_position_in_grid]])
{
    half h = as_type<half>(ushort(16u * t));
    half4 v = tanh(half4(h, -h, h, -h));
    out[2 * t + 0] = as_type<uint>(fast::tanh(float(h)));
    out[2 * t + 1] = uint(as_type<ushort>(tanh(h))) | uint(as_type<ushort>(v.x)) << 16;
}
