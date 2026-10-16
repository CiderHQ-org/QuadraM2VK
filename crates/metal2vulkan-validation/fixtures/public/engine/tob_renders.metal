// Owned fixture: the functions a Metal probe's render cases (`r.*`) drew with on the M3 Max and an
// M4, so each pass of the runtime's lane asks the question the probe measured: a rectangle and a
// triangle from a buffer's corners, a solid color from a buffer, and fragment functions that read,
// sample and write a 2D texture or a texture buffer of 4x4 texels; and the kernel of its cases that
// write a texture's bytes two ways (`k.two_ways_*`). Written for this repository; no third-party
// source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct TobVertex {
    float4 position [[position]];
};

// A rectangle, two triangles of six vertices, from corner r.xy to corner r.zw.
vertex TobVertex tob_rect(uint vid [[vertex_id]], constant float4 &r [[buffer(0)]]) {
    bool right = vid == 1u || vid == 3u || vid == 4u;
    bool up = vid == 2u || vid == 4u || vid == 5u;
    TobVertex out;
    out.position = float4(mix(r.xy, r.zw, float2(right ? 1.0 : 0.0, up ? 1.0 : 0.0)), 0.0, 1.0);
    return out;
}

// The triangle of the points p[0], p[1] and p[2].
vertex TobVertex tob_triangle(uint vid [[vertex_id]], constant float2 *p [[buffer(0)]]) {
    TobVertex out;
    out.position = float4(p[vid], 0.0, 1.0);
    return out;
}

fragment float4 tob_solid(TobVertex in [[stage_in]], constant float4 &c [[buffer(0)]]) {
    return c;
}

// The texel of t under the pixel.
fragment float4 tob_read(TobVertex in [[stage_in]], texture2d<float, access::read> t [[texture(0)]]) {
    return t.read(uint2(in.position.xy));
}

// t sampled at the pixel's center.
fragment float4 tob_sample(TobVertex in [[stage_in]], texture2d<float> t [[texture(0)]],
                           sampler s [[sampler(0)]]) {
    return t.sample(s, in.position.xy / float2(t.get_width(), t.get_height()));
}

// Texel 4y + x of the texture buffer t under pixel (x, y).
fragment float4 tob_read_buffer(TobVertex in [[stage_in]],
                                texture_buffer<float, access::read> t [[texture(0)]]) {
    uint2 p = uint2(in.position.xy);
    return t.read(p.y * 4u + p.x);
}

// Writes (x / 8, y / 8, 0.75, 1) to t's texel under pixel (x, y); returns opaque black.
fragment float4 tob_write(TobVertex in [[stage_in]], texture2d<float, access::write> t [[texture(0)]]) {
    uint2 p = uint2(in.position.xy);
    t.write(float4(0.125 * float(p.x), 0.125 * float(p.y), 0.75, 1.0), p);
    return float4(0.0, 0.0, 0.0, 1.0);
}

// The same into texel 4y + x of the texture buffer t.
fragment float4 tob_write_buffer(TobVertex in [[stage_in]],
                                 texture_buffer<float, access::write> t [[texture(0)]]) {
    uint2 p = uint2(in.position.xy);
    t.write(float4(0.125 * float(p.x), 0.125 * float(p.y), 0.75, 1.0), p.y * 4u + p.x);
    return float4(0.0, 0.0, 0.0, 1.0);
}

// One thread writes (1, 0, 0, 1) to t's texel (0, 0), and 0x5a, 0x5b, 0x5c and 0x5d to the bytes
// b[off] to b[off + 3].
kernel void tob_two_ways(texture2d<float, access::write> t [[texture(0)]],
                         device uchar *b [[buffer(0)]], constant uint &off [[buffer(1)]],
                         uint2 g [[thread_position_in_grid]]) {
    if (g.x == 0u && g.y == 0u) {
        t.write(float4(1.0, 0.0, 0.0, 1.0), uint2(0u, 0u));
        b[off] = uchar(0x5a);
        b[off + 1u] = uchar(0x5b);
        b[off + 2u] = uchar(0x5c);
        b[off + 3u] = uchar(0x5d);
    }
}
