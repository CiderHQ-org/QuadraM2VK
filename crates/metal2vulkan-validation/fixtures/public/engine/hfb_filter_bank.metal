// Owned fixture: a large `half` fragment function, for the size of its translation and the time a
// device's compiler takes to make its fragment library. A pass mode (`HfbParams::mode`) picks one
// of 32 filters in a switch; each filter loops over its taps, eight samples of a texture through a
// runtime sampler an iteration, accumulated by `half` fused multiply-adds, and then runs chains of
// `half` arithmetic (products kept apart, sums, differences, clamps, floors, maxima). Every filter
// differs in its constants, so no compiler merges two. `hfb_quad` draws the full-target triangle
// the fragment function runs over.
//
// Written for this repository; no third-party source.
#include <metal_stdlib>
using namespace metal;

struct HfbIn {
    float4 position [[position]];
    float2 uv;
};

struct HfbParams {
    uint mode;
    uint taps;
    float2 texel;
    half4 tint;
    half4 lift;
    half gain;
    half bias;
};

// The triangle over the whole target, its texture coordinates over the unit square.
vertex HfbIn hfb_quad(uint vid [[vertex_id]]) {
    float2 corner = float2(vid == 1u ? 2.0f : 0.0f, vid == 2u ? 2.0f : 0.0f);
    HfbIn out;
    out.position = float4(corner * 2.0f - 1.0f, 0.0f, 1.0f);
    out.uv = corner;
    return out;
}

// One tap: the sample at `uv + o * texel` weighted by `w` into `acc`.
#define HFB_TAP(acc, o, w) acc = fma(src.sample(smp, uv + (o) * p.texel), half4(w), acc)

// A chain of `half` arithmetic on `v`: the product `t0` is used twice, so it stays its own
// rounding; `t1 * x + y` fuses.
#define HFB_STAGE(v, k, s)                                                        \
    {                                                                             \
        half4 t0 = v * v;                                                         \
        half4 t1 = t0 * v;                                                        \
        v = v - t1 * half(0.1h + half(k & 3) * 0.01h + half(s) * 0.003h)          \
            + t0 * half(0.05h + half(s) * 0.002h);                                \
        v = clamp(v * p.gain + p.bias, half4(-2.0h), half4(2.0h + half(k & 1)));  \
        v = floor(v * half(16 + s)) * half(1.0h / half(16 + s)) + t0 * 0.001h;    \
        v = max(v, t0 - half4(1.0h)) * p.tint + p.lift * half(0.25h * half(s));   \
        v = v - t0 * t1 + t0;                                                     \
    }

// Filter `k`: its loop of taps, then three chains and one more sample.
#define HFB_CASE(k)                                                               \
    case k: {                                                                     \
        half4 acc = half4(0.0h);                                                  \
        for (uint i = 0; i < p.taps; i++) {                                       \
            float d = float(i) + 0.5f + 0.125f * float(k % 5);                    \
            HFB_TAP(acc, float2(d, 0.0f), 0.0625h + half(k) * 0.001h);            \
            HFB_TAP(acc, float2(-d, 0.0f), 0.0625h - half(k) * 0.0007h);          \
            HFB_TAP(acc, float2(0.0f, d), 0.05h + half(k & 7) * 0.002h);          \
            HFB_TAP(acc, float2(0.0f, -d), 0.05h - half(k & 3) * 0.001h);         \
            HFB_TAP(acc, float2(d, d), 0.03h + half(k) * 0.0005h);                \
            HFB_TAP(acc, float2(-d, d), 0.03h);                                   \
            HFB_TAP(acc, float2(d, -d), 0.03h - half(k) * 0.0003h);               \
            HFB_TAP(acc, float2(-d, -d), 0.03h + half(k & 1) * 0.004h);           \
            acc = clamp(acc * p.gain + p.bias, half4(0.0h), half4(4.0h));         \
        }                                                                         \
        HFB_STAGE(acc, k, 1)                                                      \
        HFB_STAGE(acc, k, 2)                                                      \
        HFB_STAGE(acc, k, 3)                                                      \
        HFB_TAP(acc, float2(0.5f * float(k), 0.25f), 0.125h + half(k) * 0.002h);  \
        c = acc;                                                                  \
    } break;

fragment half4 hfb_filter_bank(HfbIn in [[stage_in]],
                               texture2d<half> src [[texture(0)]],
                               sampler smp [[sampler(0)]],
                               constant HfbParams &p [[buffer(0)]]) {
    float2 uv = in.uv;
    half4 c = half4(0.0h);
    switch (p.mode & 31u) {
        HFB_CASE(0)
        HFB_CASE(1)
        HFB_CASE(2)
        HFB_CASE(3)
        HFB_CASE(4)
        HFB_CASE(5)
        HFB_CASE(6)
        HFB_CASE(7)
        HFB_CASE(8)
        HFB_CASE(9)
        HFB_CASE(10)
        HFB_CASE(11)
        HFB_CASE(12)
        HFB_CASE(13)
        HFB_CASE(14)
        HFB_CASE(15)
        HFB_CASE(16)
        HFB_CASE(17)
        HFB_CASE(18)
        HFB_CASE(19)
        HFB_CASE(20)
        HFB_CASE(21)
        HFB_CASE(22)
        HFB_CASE(23)
        HFB_CASE(24)
        HFB_CASE(25)
        HFB_CASE(26)
        HFB_CASE(27)
        HFB_CASE(28)
        HFB_CASE(29)
        HFB_CASE(30)
        HFB_CASE(31)
    }
    return c;
}
