// Five uncovered fragment-derivative families: `air.fwidth.v3f32` (6 corpus sources),
// `air.fwidth.v4f32` (2), `air.fwidth.v2f16` (2), `air.dfdx.v3f16` (2) and `air.dfdy.v3f16` (2) --
// 14 sources.
//
// A derivative is the one quantity a fragment case cannot author an input for: it is the difference
// between two lanes of a 2x2 quad, so it depends on the rasterizer rather than on any buffer. Two
// things make it exact anyway.
//
// The first is `[[position]]`, which is `gl_FragCoord` and not a harness varying: at pixel (px, py)
// it is exactly (px + 0.5, py + 0.5, z, 1/w), so `pos.xy - 0.5` is the integer pixel index with no
// interpolation in it. The second is that every differentiated expression here is LINEAR in that
// index, so its derivative is a constant the shader states in its own coefficients and coarse and
// fine derivatives cannot disagree. Every coefficient is a small integer, so every value and every
// difference is exact in binary32 and in binary16 alike -- the largest half any lane holds is 105.
//
// The thirteen non-zero answers are thirteen DIFFERENT integers -- 1, 2, 3, 4, 5, 6, 7, 9, 10, 13,
// 14, 15, -16 -- so a lowering that crossed two components, or took an x derivative for a y one,
// lands a number that belongs to a different lane of a different call rather than a plausible one.
// The two zeros are the x derivative of a pure-y term and the y derivative of a pure-x term, which
// are the two answers a lowering that ignored the axis cannot produce.
//
// Sixteen answers do not fit in one fragment's four channels, so the quad index selects which four
// this pixel writes. `uint(px) >> 1` is constant across a 2x2 quad, so the selection cannot make the
// quad divergent, and all five derivative calls are made before it in any case. The last channel is
// `dfdx.x + dfdy.y` across the two half calls, which is 17 only if both landed.
#include <metal_stdlib>
using namespace metal;

fragment float4 fragment_quad_derivatives(float4 pos [[position]])
{
    float px = pos.x - 0.5f;
    float py = pos.y - 0.5f;

    float3 a = float3(px, 2.0f * px + 3.0f * py, 7.0f * py);
    float3 wa = fwidth(a);                                    // (1, 5, 7)

    float4 b = float4(4.0f * px, 9.0f * py, px + py, 11.0f * px - 2.0f * py);
    float4 wb = fwidth(b);                                    // (4, 9, 2, 13)

    half2 c = half2(half(6.0f * px), half(8.0f * px + 2.0f * py));
    half2 wc = fwidth(c);                                     // (6, 10)

    half3 d = half3(half(3.0f * px), half(14.0f * py), half(15.0f * px - 16.0f * py));
    half3 dx = dfdx(d);                                       // (3, 0, 15)
    half3 dy = dfdy(d);                                       // (0, 14, -16)

    float4 q0 = float4(wa.x, wa.y, wa.z, wb.x);
    float4 q1 = float4(wb.y, wb.z, wb.w, float(wc.x));
    float4 q2 = float4(float(wc.y), float(dx.x), float(dx.y), float(dx.z));
    float4 q3 = float4(float(dy.x), float(dy.y), float(dy.z), float(dx.x) + float(dy.y));

    uint qx = uint(px) >> 1;
    return (qx == 0u) ? q0 : (qx == 1u) ? q1 : (qx == 2u) ? q2 : q3;
}
