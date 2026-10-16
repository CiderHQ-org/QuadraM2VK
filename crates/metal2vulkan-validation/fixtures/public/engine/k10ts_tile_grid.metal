// Owned fixture: a tile function over a render area of several tiles,
// some of them partial, with fewer threads per tile than cells. It measures what a tile dispatch
// gives a tile function: which pixel each imageblock coordinate is, what each builtin reads, and
// whether a cell past the render area holds a write until the dispatch ends.
//
// An `r32float` and two `r32uint` attachments are the implicit imageblock (a float attachment
// comes first because the oracle's coverage pass writes a `half4` to color 0). Each thread owns
// the 2x2 cells at `2 * tid`: it reads `f` and `v` there and writes `f * 2 + 1` and
// `v * 3 + k` (k = 1 + x % 2 + 2 * (y % 2) of the cell), so the render targets show which pixel
// each cell is and that each was read and written once, and it writes `tag` there from its own
// builtins alone (never from a value read), so every cell of the tile, past the render area too,
// holds a defined tag. After the imageblock barrier each
// thread reads the tag of the cell mirrored through the tile's centre, which another thread
// wrote, and records it beside its builtins in its own 12-word record of `out`. The record's
// slot assumes `threadgroup_position_in_grid` is the tile and that a row has at most 4 tiles;
// a slot past the buffer is skipped. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Cells {
    float f [[color(0)]];
    uint v [[color(1)]];
    uint tag [[color(2)]];
};

static uint cell_tag(ushort2 tg, ushort2 c)
{
    return 0xA0000000u | (uint(tg.y) << 20) | (uint(tg.x) << 16) | (uint(c.y) << 8) | uint(c.x);
}

kernel void k10ts_tile_grid(
    imageblock<Cells, imageblock_layout_implicit> block,
    device uint *out [[buffer(0)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort2 tg [[threadgroup_position_in_grid]],
    uint2 gid [[thread_position_in_grid]],
    ushort2 tpt [[threads_per_threadgroup]],
    uint2 tpg [[threads_per_grid]],
    ushort tidx [[thread_index_in_threadgroup]],
    uint layer [[render_target_array_index]])
{
    ushort2 c0 = tid * ushort2(2);
    ushort2 c1 = c0 + ushort2(1, 0);
    ushort2 c2 = c0 + ushort2(0, 1);
    ushort2 c3 = c0 + ushort2(1, 1);
    Cells a0 = block.read(c0);
    Cells a1 = block.read(c1);
    Cells a2 = block.read(c2);
    Cells a3 = block.read(c3);
    block.write(Cells{a0.f * 2.0f + 1.0f, a0.v * 3u + 1u, cell_tag(tg, c0)}, c0);
    block.write(Cells{a1.f * 2.0f + 1.0f, a1.v * 3u + 2u, cell_tag(tg, c1)}, c1);
    block.write(Cells{a2.f * 2.0f + 1.0f, a2.v * 3u + 3u, cell_tag(tg, c2)}, c2);
    block.write(Cells{a3.f * 2.0f + 1.0f, a3.v * 3u + 4u, cell_tag(tg, c3)}, c3);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    ushort2 last = ushort2(block.get_width() - 1, block.get_height() - 1);
    uint mirrored = block.read(last - c0).tag;
    uint slot = (uint(tg.y) * 4u + uint(tg.x)) * 64u + uint(tidx);
    if (slot < 512u) {
        device uint *r = out + slot * 12u;
        r[0] = uint(tid.x) | (uint(tid.y) << 16);
        r[1] = uint(tg.x) | (uint(tg.y) << 16);
        r[2] = gid.x;
        r[3] = gid.y;
        r[4] = uint(tpt.x) | (uint(tpt.y) << 16);
        r[5] = tpg.x;
        r[6] = tpg.y;
        r[7] = uint(tidx);
        r[8] = layer;
        r[9] = uint(block.get_width()) | (uint(block.get_height()) << 16);
        r[10] = mirrored;
        r[11] = 0xC0DE0000u | slot;
    }
}
