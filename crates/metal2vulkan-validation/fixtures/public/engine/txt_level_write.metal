// Owned fixture: a write to a `half` storage texture at a level read from a
// buffer, with 16-bit coordinates and level, as mip generators spell it
// (`air.write_texture_2d.i16.v4f16`). A storage access reaches its view's base level only, so the
// slot is bound as its level views and the write indexes them by the level; a level the texture
// lacks drops the write. Thread (x, y) writes (x, y, 1, 0.5) at texel (x, y) of level `lod`. A
// case passes level 0, which every texture has. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void txt_level_write(texture2d<half, access::write> t [[texture(0)]],
                            constant ushort &lod [[buffer(0)]],
                            ushort2 gid [[thread_position_in_grid]]) {
    t.write(half4(half(gid.x), half(gid.y), 1.0h, 0.5h), gid, lod);
}
