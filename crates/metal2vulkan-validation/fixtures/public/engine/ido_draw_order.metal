// Owned fixture: the order of framebuffer fetches across draws. Two fragment functions read color
// attachment 0 (an unsigned integer) and write it back through two affine maps that do not
// commute, `3c + 1` and `5c + 3` (wrapping), so a pass of their draws leaves in each pixel a value
// that holds only where each draw read what the draw before it wrote: a read of an older value, or
// two draws in another order, leaves another. `ido_full` draws one fullscreen triangle, and
// `ido_samples` copies each sample of a multisampled result into a buffer, `4 (y w + x) + s`.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct IdoVertex {
    float4 position [[position]];
};

vertex IdoVertex ido_full(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));
    IdoVertex out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fragment uint ido_triple(IdoVertex in [[stage_in]], uint c [[color(0)]]) {
    return c * 3u + 1u;
}

fragment uint ido_quintuple(IdoVertex in [[stage_in]], uint c [[color(0)]]) {
    return c * 5u + 3u;
}

kernel void ido_samples(texture2d_ms<uint, access::read> t [[texture(0)]],
                        device uint *out [[buffer(0)]],
                        uint2 g [[thread_position_in_grid]]) {
    if (g.x >= t.get_width() || g.y >= t.get_height()) {
        return;
    }
    uint base = 4u * (g.y * t.get_width() + g.x);
    out[base] = t.read(g, 0u).x;
    out[base + 1u] = t.read(g, 1u).x;
    out[base + 2u] = t.read(g, 2u).x;
    out[base + 3u] = t.read(g, 3u).x;
}
