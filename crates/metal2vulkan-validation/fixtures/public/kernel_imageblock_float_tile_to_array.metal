// The float form of `kernel_imageblock_tile_to_texture_array`'s whole-tile slice copy.
//
// `air.write_imageblock_slice_to_texture_2d_array.i16.v4f32` -- the last case-less
// imageblock symbol in the corpus (1 source). It is a distinct arm from the `.v4f16`
// one already covered: `imageblock_slice_texel_type` reads the texel type off the
// intrinsic's dotted suffix, and the f32 v4 branch is the one that hands back the
// module's own `v4float` instead of building a half vector.
//
// Every component is chosen to be exact in binary32 and NOT representable in binary16
// (`1 + n/1024` needs 11 significand bits at exponent 0, `5 + n/1024` needs 13), so a
// lowering that routed the f32 texel through the half path -- or that picked the texel
// type off the pointee rather than the name -- rounds every lane and fails, instead of
// agreeing by accident the way a half-exact fill would.
//
// The second write slices `aux`, the cell's SECOND member, at byte offset 16. The
// imageblock slice pointer names the FIELD and not the cell, so this is what says the
// copy walks the cell stride from a non-zero field offset rather than from the cell
// base -- a lowering that reset the origin to the cell writes `colour` into `auxOut`.
//
// Each texture is written at exactly one layer -- colour at 2, aux at 0 -- and the
// layers neither call names keep a fill no cell can produce, so "the tile went where it
// was addressed, and nowhere else" stays a thing the evidence distinguishes.
#include <metal_stdlib>
using namespace metal;

struct FCell {
    float4 colour;
    float4 aux;
};

kernel void kernel_imageblock_float_tile_to_array(
    texture2d_array<float, access::write> colourOut [[texture(0)]],
    texture2d_array<float, access::write> auxOut [[texture(1)]],
    imageblock<FCell, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 gid [[thread_position_in_grid]])
{
    threadgroup_imageblock FCell *cell = img.data(lid);
    const float x = float(lid.x) * (1.0f / 1024.0f);
    const float y = float(lid.y) * (1.0f / 1024.0f);
    const float n = float(lid.x + 16 * lid.y) * (1.0f / 1024.0f);
    cell->colour = float4(1.0f + x, 1.0f + y, 1.0f - x, 1.0f - y);
    cell->aux = float4(2.0f + n, 3.0f + n, 4.0f + n, 5.0f + n);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0, 0))) {
        colourOut.write(img.slice(cell->colour), gid, 2);
        auxOut.write(img.slice(cell->aux), gid, 0);
    }
}
