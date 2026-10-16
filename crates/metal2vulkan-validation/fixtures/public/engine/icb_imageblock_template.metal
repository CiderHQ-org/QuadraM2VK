// Owned fixture: the compute template of commands whose imageblock size a
// kernel encodes. Every thread stages a mark of its position in its threadgroup in its own cell of
// the explicit imageblock, then, after the imageblock barrier, reads its right-hand neighbour's
// (wrapping at the threadgroup's width, so only cells the threadgroup wrote are read, whatever
// the imageblock's size) and reports it with the imageblock's extent:
//     out[gid.y * grid.x + gid.x] = (0x10000 | ((lid.x + 1) % tg.x) | lid.y << 8, width, height,
//                                    gid.x | gid.y << 8)
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Cell {
    uint value;
};

kernel void icb_imageblock_template(
    imageblock<Cell, imageblock_layout_explicit> img,
    device uint4 *out [[buffer(0)]],
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 tg [[threads_per_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]])
{
    threadgroup_imageblock Cell *mine = img.data(lid);
    mine->value = 0x10000u | uint(lid.x) | (uint(lid.y) << 8);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    uint other = img.data(ushort2((lid.x + 1) % tg.x, lid.y))->value;
    out[gid.y * grid.x + gid.x] =
        uint4(other, uint(img.get_width()), uint(img.get_height()), gid.x | (gid.y << 8));
}
