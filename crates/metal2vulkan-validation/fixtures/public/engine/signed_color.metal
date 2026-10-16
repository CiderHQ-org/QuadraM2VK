// Owned fixture: fragment color outputs to signed-integer attachments.
// A render pipeline whose color attachment has a signed-integer pixel format compiles its
// fragment function with that attachment in `LoweringInputs::signed_color_attachments`, so a
// 32-bit integer output to it is declared `int` and stored by `OpBitcast`. The `int4` lanes are
// chosen so a float round trip or a saturating conversion would move them. Loop-free.
#include <metal_stdlib>
using namespace metal;

// A fullscreen triangle.
vertex float4 signed_color_v(uint vid [[vertex_id]]) {
    float2 uv = float2(float((vid << 1) & 2u), float(vid & 2u));
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

// The whole return is the signed-integer color.
fragment int4 signed_color_f(float4 pos [[position]]) {
    int x = int(pos.x);
    return int4(16777217 + x, -16777217 - x, 2147483647 - x, -2147483647 - 1 + x);
}

// One output of each class: float, signed, and unsigned integer; the signed one is offset by a
// buffer value.
struct SignedTargets {
    float4 color [[color(0)]];
    int4 ids [[color(1)]];
    uint2 marks [[color(2)]];
};

fragment SignedTargets signed_targets_f(float4 pos [[position]],
                                        constant int4 &bias [[buffer(0)]]) {
    SignedTargets t;
    int x = int(pos.x);
    t.color = float4(pos.xy, 0.0, 1.0);
    t.ids = int4(-x, x - 1000000, -1, 7) + bias;
    t.marks = uint2(uint(x), 0xFFFFFFFFu);
    return t;
}

// A narrow signed output: which extension fills a wider attachment's lanes is not established.
fragment short4 narrow_signed_color_f(float4 pos [[position]]) {
    return short4(-1, 2, -3, short(pos.x));
}
