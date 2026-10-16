// Owned fixture: the extent queries of a kernel's explicit imageblock
// (`access.imageblock.extent`), as `kernel_imageblock_staging`'s `imageblock_neighbours`, alone in
// its module so a case can name it. Every thread stages its own cell, then, after the imageblock
// barrier, reads its right-hand neighbour's (wrapping at the imageblock's width) and reports the
// extent the queries answer. Dispatched with an imageblock the size of the threadgroup (W x H):
//     out[y * W + x] = (bits(float((x+1)%W) + 100 * y), ((x+1)%W) | (y << 16), W, H)
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Cell {
    float value;
    ushort2 at;
};

kernel void a4_imageblock_extent(imageblock<Cell, imageblock_layout_explicit> img,
                                 device uint4 *out [[buffer(0)]],
                                 ushort2 lid [[thread_position_in_threadgroup]],
                                 ushort2 tg [[threads_per_threadgroup]]) {
    threadgroup_imageblock Cell *mine = img.data(lid);
    mine->value = float(lid.x) + 100.0f * float(lid.y);
    mine->at = lid;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    ushort w = img.get_width();
    ushort h = img.get_height();
    threadgroup_imageblock Cell *other = img.data(ushort2((lid.x + 1) % w, lid.y));
    uint i = uint(lid.y) * uint(tg.x) + uint(lid.x);
    out[i] = uint4(as_type<uint>(other->value), uint(other->at.x) | (uint(other->at.y) << 16),
                   uint(w), uint(h));
}
