// Owned fixture: the functions a Metal probe's passes ran (tile dispatches over 2D textures whose
// texels lie in buffers), so each pass of the runtime's lane asks the question the probe measured.
// The passes are 32x32 in 16x16 tiles; a thread or a fragment at pixel p whose coordinates are
// multiples of 8 works on texel c = p / 8 of a 4x4 texture. A fullscreen triangle; fragment
// functions that write a texture's texel, write a texel's bytes through a buffer, and read a
// texel's bytes or the texel into `o`; tile functions that copy a texel, its channels reversed,
// from a bound texture or one an argument buffer holds; and one that writes a texel and other bytes
// of its buffer (two ways). Written for this repository; no third-party source. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct CfbTsVertex {
    float4 position [[position]];
};

static bool cfb_ts_corner(uint2 p) {
    return (p.x & 7u) == 0u && (p.y & 7u) == 0u;
}

// v's channels as unorm bytes, red the lowest.
static uint cfb_ts_pack(float4 v) {
    uint4 u = uint4(round(saturate(v) * 255.0));
    return u.x | (u.y << 8) | (u.z << 16) | (u.w << 24);
}

vertex CfbTsVertex cfb_ts_full(uint vid [[vertex_id]]) {
    CfbTsVertex out;
    out.position = float4(vid == 1u ? 3.0 : -1.0, vid == 2u ? 3.0 : -1.0, 0.0, 1.0);
    return out;
}

// Texel c of t becomes (c.x / 8, c.y / 8, 0.75, 1).
fragment float4 cfb_ts_write(CfbTsVertex in [[stage_in]],
                             texture2d<float, access::write> t [[texture(0)]]) {
    uint2 p = uint2(in.position.xy);
    if (cfb_ts_corner(p)) {
        uint2 c = p / 8u;
        t.write(float4(0.125 * float(c.x), 0.125 * float(c.y), 0.75, 1.0), c);
    }
    return float4(0.0);
}

// Word at.x + c.y * at.y + c.x of b (texel c's bytes) becomes 0xA5330000 | c.y << 8 | c.x.
fragment float4 cfb_ts_write_bytes(CfbTsVertex in [[stage_in]], device uint *b [[buffer(0)]],
                                   constant uint2 &at [[buffer(1)]]) {
    uint2 p = uint2(in.position.xy);
    if (cfb_ts_corner(p)) {
        uint2 c = p / 8u;
        b[at.x + c.y * at.y + c.x] = 0xA5330000u | (c.y << 8) | c.x;
    }
    return float4(0.0);
}

// o[c.y * 4 + c.x] = word at.x + c.y * at.y + c.x of b.
fragment float4 cfb_ts_read_bytes(CfbTsVertex in [[stage_in]], const device uint *b [[buffer(0)]],
                                  constant uint2 &at [[buffer(1)]], device uint *o [[buffer(2)]]) {
    uint2 p = uint2(in.position.xy);
    if (cfb_ts_corner(p)) {
        uint2 c = p / 8u;
        o[c.y * 4u + c.x] = b[at.x + c.y * at.y + c.x];
    }
    return float4(0.0);
}

// o[c.y * 4 + c.x] = texel c of t, packed.
fragment float4 cfb_ts_read_texture(CfbTsVertex in [[stage_in]],
                                    texture2d<float, access::read> t [[texture(0)]],
                                    device uint *o [[buffer(2)]]) {
    uint2 p = uint2(in.position.xy);
    if (cfb_ts_corner(p)) {
        uint2 c = p / 8u;
        o[c.y * 4u + c.x] = cfb_ts_pack(t.read(c));
    }
    return float4(0.0);
}

// Texel c of dst becomes texel c of src, its channels reversed.
kernel void cfb_ts_copy(texture2d<float, access::read> src [[texture(0)]],
                        texture2d<float, access::write> dst [[texture(1)]],
                        ushort2 g [[thread_position_in_grid]]) {
    uint2 p = uint2(g);
    if (cfb_ts_corner(p)) {
        uint2 c = p / 8u;
        dst.write(src.read(c).abgr, c);
    }
}

struct CfbTsHeld {
    texture2d<float, access::read> src;
};

// As cfb_ts_copy, src held in an argument buffer.
kernel void cfb_ts_copy_held(constant CfbTsHeld &h [[buffer(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             ushort2 g [[thread_position_in_grid]]) {
    uint2 p = uint2(g);
    if (cfb_ts_corner(p)) {
        uint2 c = p / 8u;
        dst.write(h.src.read(c).abgr, c);
    }
}

// One thread writes (1, 0, 0, 1) to t's texel (0, 0), and 0x5a, 0x5b, 0x5c and 0x5d to the bytes
// b[off] to b[off + 3].
kernel void cfb_ts_two_ways(texture2d<float, access::write> t [[texture(0)]],
                            device uchar *b [[buffer(0)]], constant uint &off [[buffer(1)]],
                            ushort2 g [[thread_position_in_grid]]) {
    if (g.x == 0 && g.y == 0) {
        t.write(float4(1.0, 0.0, 0.0, 1.0), uint2(0u, 0u));
        b[off] = uchar(0x5a);
        b[off + 1u] = uchar(0x5b);
        b[off + 2u] = uchar(0x5c);
        b[off + 3u] = uchar(0x5d);
    }
}
