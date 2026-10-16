// Owned fixture: a tile function's explicit imageblock aliased onto the
// implicit one, whose members' C offsets (0, 4, 8, 16) differ from a packed layout of r16Float,
// r32Float, rgba16Float, and r16Float attachments (0, 2, 6, 14). Each thread records the bits of
// every member of its cell, so the record shows where each attachment's texel sits in the cell
// (member i holds attachment i's texel bits at the member's offset). Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Mixed {
    half a;
    float b;
    half4 c;
    half d;
};

kernel void k10ts_alias_probe(
    imageblock<Mixed, imageblock_layout_explicit> block [[alias_implicit_imageblock]],
    device uint *out [[buffer(0)]],
    ushort2 tid [[thread_position_in_threadgroup]])
{
    threadgroup_imageblock Mixed *cell = block.data(tid);
    device uint *r = out + (uint(tid.y) * 16u + uint(tid.x)) * 5u;
    half4 c = cell->c;
    r[0] = uint(as_type<ushort>(cell->a));
    r[1] = as_type<uint>(cell->b);
    r[2] = uint(as_type<ushort>(c.x)) | (uint(as_type<ushort>(c.y)) << 16);
    r[3] = uint(as_type<ushort>(c.z)) | (uint(as_type<ushort>(c.w)) << 16);
    r[4] = uint(as_type<ushort>(cell->d));
}
