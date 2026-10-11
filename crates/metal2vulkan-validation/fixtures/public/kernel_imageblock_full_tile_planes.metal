// A whole 32x32 threadgroup's imageblock, copied out as two planes of different width.
//
// `air.write_imageblock_slice_to_texture_2d.i16.f16` (24 corpus sources) and
// `air.write_imageblock_slice_to_texture_2d.i16.v2f16` (15) -- 39 sources, and the two symbols the
// five corpus kernels use whose 8-byte luma/chroma cell this fixture is modelled on.
//
// 32x32 is the largest tile a threadgroup can have: one cell per thread at Metal's 1024-thread
// maximum. Neither slice names a size, so both regions are the threadgroup extent and the copy is
// of all 1024 cells -- more than the flat 512-cell array this translator used to allocate, which
// refused the module outright. An 8-byte cell puts 1024 of them at 8 KiB, half of the threadgroup
// memory Vulkan guarantees.
//
// Every thread stages a value that names its own cell and nothing else: the luma is
// `x + 32 * y`, which is distinct across all 1024 cells, and the chroma is `(x/2, y/4)`, which
// separates the two axes so a copy that transposed them or walked one row stride short would land
// values that exist elsewhere in the tile rather than values that exist nowhere. All of them are
// integers or exact binary fractions below 1024, so every one is exact in binary16 and the
// comparison needs no tolerance.
//
// The two planes sit at different offsets in the cell (0 and 4) and have different widths, so a
// slice that took its stride from the cell rather than from the plane cannot satisfy both.
#include <metal_stdlib>
using namespace metal;

struct TileCell {
    half  luma;
    half2 chroma;
};

kernel void kernel_imageblock_full_tile_planes(
    texture2d<half, access::write> lumaOut [[texture(0)]],
    texture2d<half, access::write> chromaOut [[texture(1)]],
    imageblock<TileCell, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 gid [[thread_position_in_grid]])
{
    threadgroup_imageblock TileCell *cell = img.data(lid);
    cell->luma = half(lid.x) + half(lid.y) * 32.0h;
    cell->chroma = half2(half(lid.x) * 0.5h, half(lid.y) * 0.25h);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0, 0))) {
        lumaOut.write(img.slice(cell->luma), gid);
        chromaOut.write(img.slice(cell->chroma), gid);
    }
}
