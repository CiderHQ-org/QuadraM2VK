// Owned fixture: `a1_sign_and_division` without fast math (`-fno-fast-math`: `sign`
// carries no `nsz`, so the exact sequence, which keeps -0, is the lowering, and the module
// records `air.compile.fast_math_disable`, so Metal's division is its precise one). The same
// program; the pair separates what fast math changes in the zero's sign and the division range.
// metal-fast-math: off
//
// Thread i reads `in[i]` = float patterns (w, lr, s, g), `div[i]` = float patterns (n, d), and
// `hin[i]` = half patterns (x, n, d, 0), and writes four words: `sign(w)` (`air.sign.f32`),
// `w - lr * (sign(w) * s + g)` (the L1 regularizer step of d72f0f18), `n / d` (`fdiv float`),
// and the half `sign(x) | (n / d) << 16` (`air.sign.f16`, `fdiv half`). The sign lanes are
// signed zeros, denormals, ones, halves, infinities, NaNs, and the largest finite, with the
// regularizer at lr = 0.5, s = 1, g = +0 and a few other zero signs; the division lanes put
// divisors at and past 2^126 (1.5 * 2^126, 2^127, 1.5 * 2^127, the largest finite, negated),
// denormal divisors, zeros, and infinities under dividends whose exact quotients are
// representable (the half lanes do the same at 2^14, 2^15, 65504, and the half denormals).
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a1_sign_and_division_precise(
    device const uint4 *in [[buffer(0)]],
    device const uint2 *div [[buffer(1)]],
    device const ushort4 *hin [[buffer(2)]],
    device uint4 *out [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    float4 r = as_type<float4>(in[i]);
    float2 q = as_type<float2>(div[i]);
    half4 h = as_type<half4>(hin[i]);
    float w = r.x;
    out[i] = uint4(as_type<uint>(sign(w)), as_type<uint>(w - r.y * (sign(w) * r.z + r.w)),
                   as_type<uint>(q.x / q.y),
                   uint(as_type<ushort>(sign(h.x))) | uint(as_type<ushort>(h.y / h.z)) << 16);
}
