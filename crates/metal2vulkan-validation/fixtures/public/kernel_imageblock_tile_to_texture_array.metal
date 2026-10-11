// The same whole-tile slice copy as `kernel_imageblock_full_tile_planes`, but into a chosen slice of
// a texture array.
//
// `air.write_imageblock_slice_to_texture_2d_array.i16.v4f16` (17 case-less corpus sources) and
// `air.write_imageblock_slice_to_texture_2d_array.i16.f16` (11) -- 28 sources. The array form takes
// one more operand than the 2D one, the array slice, and it is the operand nothing had pinned: a
// lowering that dropped it, or that confused it with the level operand beside it, writes a
// complete and correct-looking tile into the wrong layer.
//
// So each texture is written at exactly one layer -- colour at 2, weight at 1 -- and the layers
// neither call names keep a fill no cell can hold. Three cases read the two written layers and one
// unwritten one, which is what makes "the tile went where it was addressed, and nowhere else" a
// thing the evidence can distinguish rather than an assumption.
//
// The 16-byte cell is deliberate: 1024 of them is 16384 bytes, exactly the threadgroup memory
// Vulkan guarantees and exactly where this translator stops allocating cells. Running it is what
// makes that budget a measured limit instead of a chosen one.
//
// The colour separates its four channels (x, y, x/2, y/4) so a copy that transposed the tile or
// swizzled the texel lands a quadruple that exists nowhere; the weight is x + 32 * y, distinct
// across all 1024 cells. Every value is exact in binary16.
#include <metal_stdlib>
using namespace metal;

struct ArrCell {
    half4 colour;
    half  weight;
};

kernel void kernel_imageblock_tile_to_texture_array(
    texture2d_array<half, access::write> colourOut [[texture(0)]],
    texture2d_array<half, access::write> weightOut [[texture(1)]],
    imageblock<ArrCell, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 gid [[thread_position_in_grid]])
{
    threadgroup_imageblock ArrCell *cell = img.data(lid);
    cell->colour = half4(half(lid.x), half(lid.y), half(lid.x) * 0.5h, half(lid.y) * 0.25h);
    cell->weight = half(lid.x) + half(lid.y) * 32.0h;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0, 0))) {
        colourOut.write(img.slice(cell->colour), gid, 2);
        weightOut.write(img.slice(cell->weight), gid, 1);
    }
}
