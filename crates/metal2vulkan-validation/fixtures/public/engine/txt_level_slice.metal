// Owned fixture: an imageblock slice written to a `half` storage texture at a
// level read from a buffer, as mip-chain blurs spell it
// (`air.write_imageblock_slice_to_texture_2d.i16.v4f16` with a runtime level). Each thread stages
// (x, y, 1, 0.5) in its cell; thread (0, 0) then writes the whole imageblock's slice at texel
// (0, 0) of level `lod`, which lands cell (x, y) at texel (x, y) of that level; a level the
// texture lacks drops the slice. A case passes level 0. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Cell {
    half4 color;
};

kernel void txt_level_slice(texture2d<half, access::write> t [[texture(0)]],
                            constant ushort &lod [[buffer(0)]],
                            imageblock<Cell, imageblock_layout_explicit> img,
                            ushort2 lid [[thread_position_in_threadgroup]]) {
    threadgroup_imageblock Cell *cell = img.data(lid);
    cell->color = half4(half(lid.x), half(lid.y), 1.0h, 0.5h);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0, 0))) {
        t.write(img.slice(cell->color), ushort2(0, 0), lod);
    }
}
