// Owned fixture: rounding to `half` just past its largest finite value, without fast math
// (`-fno-fast-math`: an addition whose result overflows carries no `ninf`). The translator rounds
// every `half` result with `OpQuantizeToF16`, whose SPIR-V definition makes a value "too large to
// represent" an infinity; round to nearest even keeps everything below 65520 at 65504 (the largest
// half) and makes 65520 (a tie, 65504 being odd) and above infinite. Metal's conversion and its
// half arithmetic in that range are unmeasured.
//
// Thread t (0..4159) writes two words:
// - `half(x) | half(y) << 16` for `x = 65504 + t / 256` and `y = -x`, every `float` from 65504 to
//   65520 (t = 4096) and 63 past it, and their negations (two `fptrunc`s: y is read from its own
//   bit pattern, so the front end cannot fold its conversion into a negation of x's);
// - `(65504h + b) | (-65504h - b) << 16` for the half `b` of pattern `0x4000 + t` (every half in
//   [2, 34); the exact sum is 65520 at b = 16, t = 3072), a half `fadd` and `fsub`.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void a1_quantize_f16_overflow(
    device uint *out [[buffer(0)]],
    uint t [[thread_position_in_grid]])
{
    float x = as_type<float>(0x477fe000u + t);
    float y = as_type<float>(0xc77fe000u + t);
    half b = as_type<half>(ushort(0x4000u + t));
    const half top = 65504.0h;
    out[2 * t + 0] = uint(as_type<ushort>(half(x))) | uint(as_type<ushort>(half(y))) << 16;
    out[2 * t + 1] = uint(as_type<ushort>(top + b)) | uint(as_type<ushort>(-top - b)) << 16;
}
