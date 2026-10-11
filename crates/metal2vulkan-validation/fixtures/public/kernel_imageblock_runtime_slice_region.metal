// Five imageblock slice writes whose block extent is `[[threads_per_threadgroup]]` arithmetic -- a
// value AIR computes at runtime rather than a constant the translator can read -- asked from three
// different cells so that where the block STARTS is answered too.
//
// `air.write_imageblock_slice_to_texture_2d.i16.v4i16` (2 case-less corpus sources),
// `air.write_imageblock_slice_to_texture_2d.f16` (1) and
// `air.write_imageblock_slice_to_texture_2d.v4f32` (2) -- 5 sources, and the device half of the
// 22-source class of corpus modules whose slice region is a runtime value.
//
// The threadgroup is 8x8 and each thread owns one cell, so the imageblock is 8x8 and a lowering that
// took the block extent from the imageblock rather than from the size operand agrees with the first
// write and with nothing else. The CODE plane asks for `tgSize >> 1`, a 4x4 block, at (8, 0): a copy
// sized by the imageblock would write 8x8 there and put 48 texels of tile content over the authored
// poison, in a corner of the texture the answer never reaches. The COLOUR plane asks for the whole
// `tgSize` block at (0, 0), the one case where the two readings agree, so that a lowering copying
// nothing at all cannot be mistaken for one that reads the operand.
//
// The three WEIGHT writes are the origin question, and Metal's answer is the reason this file exists
// in the shape it does. They ask for the same field at three sizes from three different cells:
//
//   | slice pointer | size          | destination | Metal wrote        |
//   |---------------|---------------|-------------|--------------------|
//   | cell (4, 4)   | tgSize >> 1   | (0, 8)      | cells (0..3, 0..3) |
//   | cell (0, 0)   | tgSize >> 2   | (8, 8)      | cells (0..1, 0..1) |
//   | cell (5, 3)   | tgSize >> 3   | (12, 12)    | cell (0, 0)        |
//
// Not one of them started where the pointer pointed. **The block always starts at the imageblock's
// own origin; the pointer selects which FIELD of the cell is copied and its cell coordinate is
// ignored.** The 1x1 row is what rules out a rounding or clamping story: a copy anchored at the
// pointer would have answered 29 there, and it answered 0.
//
// Every cell holds a value that exists nowhere else in the tile. The colour is
// (x, y, x/2 + 1/4, y/4 + 1/2), the code is (7x - 3, 11y - 5, 100x + y, -(x + 16y)) and the weight
// is x + 8y, so a transposed, swizzled or shifted copy lands a quadruple that exists nowhere in the
// tile at all -- which is what makes the three origin rows readable as coordinates rather than as
// noise. Every float is an integer or an exact binary fraction below 16 and every half an integer
// below 64, so all are exact in their own format and the comparison needs no tolerance.
//
// The three planes carry three texel types and two coordinate types -- a `ushort2` write for the
// codes and `uint2` writes for the other two -- because the AIR symbol spells both, and a lowering
// reading the coordinate at the wrong width would place a whole block somewhere else.
//
// The grid is exactly one threadgroup: an explicit-layout imageblock with no implicit coverage takes
// the compute path, and a second tile would copy its own block to the same texture origin.
#include <metal_stdlib>
using namespace metal;

struct TileCell {
    float4 colour;
    short4 code;
    half   weight;
};

kernel void kernel_imageblock_runtime_slice_region(
    texture2d<float, access::write> colourOut [[texture(0)]],
    texture2d<short, access::write> codeOut [[texture(1)]],
    texture2d<half, access::write> weightOut [[texture(2)]],
    imageblock<TileCell, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 tgSize [[threads_per_threadgroup]])
{
    threadgroup_imageblock TileCell *mine = img.data(lid);
    float x = float(lid.x);
    float y = float(lid.y);
    mine->colour = float4(x, y, x * 0.5f + 0.25f, y * 0.25f + 0.5f);
    mine->code = short4(short(lid.x) * 7 - 3, short(lid.y) * 11 - 5,
                        short(lid.x) * 100 + short(lid.y), -(short(lid.x) + 16 * short(lid.y)));
    mine->weight = half(lid.x) + 8.0h * half(lid.y);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0, 0))) {
        threadgroup_imageblock TileCell *origin = img.data(ushort2(0, 0));
        threadgroup_imageblock TileCell *quad = img.data(ushort2(4, 4));
        colourOut.write(img.slice(origin->colour, tgSize), uint2(0, 0));
        codeOut.write(img.slice(origin->code, tgSize >> 1), ushort2(8, 0));
        threadgroup_imageblock TileCell *odd = img.data(ushort2(5, 3));
        weightOut.write(img.slice(quad->weight, tgSize >> 1), uint2(0, 8));
        weightOut.write(img.slice(origin->weight, tgSize >> 2), uint2(8, 8));
        weightOut.write(img.slice(odd->weight, tgSize >> 3), uint2(12, 12));
    }
}
