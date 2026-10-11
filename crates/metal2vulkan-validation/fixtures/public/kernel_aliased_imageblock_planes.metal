// A tile kernel whose EXPLICIT imageblock is aliased onto the implicit one, so its cell is not
// tile-local scratch but the render targets themselves.
//
// `[[alias_implicit_imageblock]]` is the source attribute that puts `air.alias_implicit_imageblock`
// on the `air.imageblock` argument node. 30 of the 14579 corpus sources carry it, in four distinct
// layouts, and this is the widest of the four: `{half4, half4, half}`, 24 bytes. That the AIR is
// byte-for-byte the corpus's shape is the point -- the loads and stores below come out as the same
// `getelementptr i8` off `air.imageblock_data` that every one of the 30 uses.
//
// The kernel measures three separate claims at once:
//
//  * **The cell is read from the attachment.** Every value written is a function of values read, so
//    a cell that started as undefined scratch answers differently.
//  * **The cell is written back to the attachment.** The comparison is made in the render targets.
//  * **Member `i` is attachment `i`.** Every output plane mixes lanes from more than one input
//    plane, so swapping two attachments, or answering the depth-typed member from the depth
//    attachment rather than from colour attachment 2, changes the answer at every texel.
//
// The arithmetic is exact and association-proof. The initial texels are
// `colour = (x, y, x + 16y, 1)`, `aux = (x + 300, y + 300, 600 + x + 16y, 7)` and
// `depth = 100 + x + 16y` -- every one an integer, every one distinct across the 256 cells, and the
// third lane of each carries `x + 16y` so a column error and a row error separate. Every result is
// an integer below 2048, which binary16 represents exactly, so no lane depends on rounding, on fast
// math, or on the order the sum in `depth` is associated:
//
//     colour' = (x + 1, y + 302, 103 + x + 16y, 2)          max 358
//     aux'    = (2x + 600, y + 4, 300 + 3x + 48y, 12)       max 1065
//     depth'  = 700 + 3x + 48y                              max 1465
#include <metal_stdlib>
using namespace metal;

struct ColorBlock {
    half4 color;
    half4 aux;
    half depth;
};

kernel void kernel_aliased_imageblock_planes(
    imageblock<ColorBlock, imageblock_layout_explicit> block [[alias_implicit_imageblock]],
    ushort2 position [[thread_position_in_threadgroup]]) {
    threadgroup_imageblock ColorBlock* cell = block.data(position);
    half4 c = cell->color;
    half4 a = cell->aux;
    half d = cell->depth;
    cell->color = half4(c.x + 1.0h, a.y + 2.0h, d + 3.0h, c.w * 2.0h);
    cell->aux = half4(a.x * 2.0h, c.y + 4.0h, d * 3.0h, a.w + 5.0h);
    cell->depth = c.z + a.z + d;
}
