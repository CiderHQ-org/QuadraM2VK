// Owned fixture: programmable blending in a multisampled render pipeline, beside `[[position]]`'s
// w. Metal runs a fragment function that reads a color attachment once per covered sample and
// reads `[[position]]` as the pixel's in each of those runs. Loop-free; nothing here runs on a GPU.
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
};

// A strip of two triangles over the viewport, at a depth and w that vary across it.
vertex Varyings multisample_fetch_vertex(uint vid [[vertex_id]]) {
    float2 p = float2(float(vid & 1u), float(vid >> 1));
    float w = 1.0 + p.x;
    Varyings out;
    out.position = float4((p * 2.0 - 1.0) * w, 0.5 * w, w);
    return out;
}

// The attachment's color, scaled by `[[position]]`'s w (1/w, as a compositor reads it).
fragment float4 multisample_fetch_depth(Varyings in [[stage_in]], float4 dst [[color(0)]]) {
    return dst * in.position.w;
}
