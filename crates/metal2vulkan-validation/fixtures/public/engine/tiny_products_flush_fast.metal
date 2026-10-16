// Owned fixture: `float` products whose exact value lies next to `2^-126`, the smallest normal,
// under fast math. Metal flushes a product whose exact value is below `2^-126` to the zero of its
// sign, even where rounding it would reach `2^-126` (tininess before rounding); a Vulkan device
// rounds first, and a flush after the rounding keeps `2^-126`. So `(1 - 2^-24) * 2^-126`, exact, is
// `+0` on Metal and `2^-126` there.
//
// Thread `t` reads the float4 operands `a = in[2t]` and `b = in[2t + 1]` and writes three float4
// rows: `a * b` as a vector multiply; the same four products lane by lane as scalar multiplies of
// scalar loads; and `dot(a, b)`, `dot(a.xy, b.xy)`, `dot(a.zw, b.zw)` and `dot(a.yzw, b.yzw)`,
// Metal's fused chains whose first product and later fused multiply-adds meet the same boundary.
// The lanes mix both signs, exponent sums of 126 and 127 around the boundary, the exact `2^-126`,
// products just above it and the representable denormal just below it, and ordinary products.
// `tiny_products_flush` asks the same without fast math. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void tiny_products_flush_fast(
    device const float4 *in [[buffer(0)]],
    device float4 *out [[buffer(1)]],
    uint t [[thread_position_in_grid]])
{
    float4 a = in[2 * t];
    float4 b = in[2 * t + 1];
    device const float *s = (device const float *)in;
    out[3 * t] = a * b;
    out[3 * t + 1] = float4(s[8 * t] * s[8 * t + 4], s[8 * t + 1] * s[8 * t + 5],
                            s[8 * t + 2] * s[8 * t + 6], s[8 * t + 3] * s[8 * t + 7]);
    out[3 * t + 2] = float4(dot(a, b), dot(a.xy, b.xy), dot(a.zw, b.zw), dot(a.yzw, b.yzw));
}
