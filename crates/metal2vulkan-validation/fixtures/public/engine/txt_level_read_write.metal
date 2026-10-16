// Owned fixture: a `read_write` texture read and written at a level read
// from a buffer: thread (x, y) reads texel (x, y) of level `lod` and writes back 2 v + 1. A read
// or write at a level the texture lacks is out of bounds (MSL 4.1 §6.13): Metal reads zero
// components (alpha 1 where the format has none) and ignores the write. A case passes level 0.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void txt_level_read_write(texture2d<float, access::read_write> t [[texture(0)]],
                                 constant uint &lod [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    float4 v = t.read(gid, lod);
    t.write(v * 2.0f + 1.0f, gid, lod);
}
