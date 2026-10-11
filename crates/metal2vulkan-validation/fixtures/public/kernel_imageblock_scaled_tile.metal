// A tile twice the threadgroup in each axis: 8x8 threads staging a 2x2 block of imageblock cells
// each, then copied out as two planes.
//
// `air.write_imageblock_slice_to_texture_2d.i16.v4f32` (7 case-less corpus sources) and
// `air.write_imageblock_slice_to_texture_2d.i16.f32` (4) -- 11 sources.
//
// Every other authored imageblock case owns one cell per thread, which is what made the threadgroup
// extent and the imageblock extent the same number and let three separate places assume it. Here
// they differ: the tile is 16x16 and the threadgroup is 8x8, so a lowering that linearised the tile
// by the threadgroup width lands cells (0,1) and (8,0) on the same index -- two threads writing each
// other's staging -- and one that took the implicit slice region from the threadgroup extent copies
// the top-left 8x8 corner of a block Metal copies whole.
//
// Neither failure is quiet here. The colour is (X, Y, X/2, Y/4) and the weight is X + 16Y at cell
// (X, Y), so every one of the 256 cells holds a value that exists nowhere else in the tile, the
// four colour channels scale differently so a transpose or a swizzle lands a quadruple that exists
// nowhere at all, and the two planes have different widths in the cell so a stride taken from the
// cell rather than the plane cannot satisfy both. A quarter-sized copy leaves 192 of the 256 texels
// at their authored poison. Every value is an integer or an exact binary fraction below 256, so all
// of them are exact in binary32 and the comparison needs no tolerance.
//
// The slice regions are implicit -- `slice(field)` with no size -- which is the form that has to
// derive the extent rather than read it, and the form 2 of the 6 corpus modules this unblocks use.
//
// The 32-byte cell puts 256 of them at 8 KiB, half the threadgroup memory Vulkan guarantees.
#include <metal_stdlib>
using namespace metal;

struct TileCell {
    float4 colour;
    float  weight;
};

static void stage(imageblock<TileCell, imageblock_layout_explicit> img, ushort2 at)
{
    threadgroup_imageblock TileCell *cell = img.data(at);
    cell->colour = float4(float(at.x), float(at.y), float(at.x) * 0.5f, float(at.y) * 0.25f);
    cell->weight = float(at.x) + 16.0f * float(at.y);
}

kernel void kernel_imageblock_scaled_tile(
    texture2d<float, access::write> colourOut [[texture(0)]],
    texture2d<float, access::write> weightOut [[texture(1)]],
    imageblock<TileCell, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 gid [[thread_position_in_grid]])
{
    ushort2 base = lid * 2;
    stage(img, base + ushort2(0, 0));
    stage(img, base + ushort2(1, 0));
    stage(img, base + ushort2(0, 1));
    stage(img, base + ushort2(1, 1));
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0, 0))) {
        threadgroup_imageblock TileCell *origin = img.data(ushort2(0, 0));
        colourOut.write(img.slice(origin->colour), gid);
        weightOut.write(img.slice(origin->weight), gid);
    }
}
