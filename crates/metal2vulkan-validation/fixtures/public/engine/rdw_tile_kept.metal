// Owned fixture: tile functions whose explicit imageblock stands alone and holds sixteen bytes a
// cell, one `uint4`, which Metal places after an `r32uint` attachment's four bytes rounded up to 8
// and then to the block's 16-byte alignment (the state's imageblock sample length is 32). Plain
// Metal keeps the words of such a block that lie below the pass's imageblock sample length rounded
// up to 8, and reads every other word as zero, a dispatch's own writes to it read back after a
// barrier included (Metal probes on an M4 under macOS 27.0.1): a pass of 20 bytes keeps words 0 and
// 1. A tile function runs one thread per pixel of its tile, its cell
// `thread_position_in_threadgroup`. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

// The cell: four words.
struct Wide {
    uint4 w;
};

// What each function reads at tile buffer 1: the render area's size and a seed.
struct Args {
    uint2 size;
    uint seed;
    uint pad;
};

// Every word of every cell: word k takes `seed + k`.
kernel void rdw_tile_kept_fill(imageblock<Wide, imageblock_layout_explicit> block,
                               constant Args &args [[buffer(1)]],
                               ushort2 t [[thread_position_in_threadgroup]])
{
    threadgroup_imageblock Wide *cell = block.data(t);
    uint s = args.seed;
    cell->w = uint4(s, s + 1u, s + 2u, s + 3u);
}

// out[4 * i + k] takes word k of pixel `i = y * size.x + x`'s cell, for the pixels inside the
// render area.
kernel void rdw_tile_kept_dump(imageblock<Wide, imageblock_layout_explicit> block,
                               device uint *out [[buffer(0)]],
                               constant Args &args [[buffer(1)]],
                               ushort2 t [[thread_position_in_threadgroup]],
                               ushort2 g [[threadgroup_position_in_grid]],
                               ushort2 n [[threads_per_threadgroup]])
{
    uint4 w = block.data(t)->w;
    uint2 p = uint2(g) * uint2(n) + uint2(t);
    if (p.x < args.size.x && p.y < args.size.y) {
        uint i = p.y * args.size.x + p.x;
        out[4u * i] = w.x;
        out[4u * i + 1u] = w.y;
        out[4u * i + 2u] = w.z;
        out[4u * i + 3u] = w.w;
    }
}

// Word k of the cell takes `seed + 16 + k`, one word at a time at the index `k` the thread's
// position picks, and is read back after an imageblock barrier (which keeps the compiler from
// forwarding the written values, as probe 7's `within` case read them), a word at a time, into
// out[4 * i + k], as `rdw_tile_kept_dump` places the pixels: a word Metal does not keep reads zero
// after this dispatch's own write.
kernel void rdw_tile_kept_fill_dump(imageblock<Wide, imageblock_layout_explicit> block,
                                    device uint *out [[buffer(0)]],
                                    constant Args &args [[buffer(1)]],
                                    ushort2 t [[thread_position_in_threadgroup]],
                                    ushort2 g [[threadgroup_position_in_grid]],
                                    ushort2 n [[threads_per_threadgroup]])
{
    threadgroup_imageblock Wide *cell = block.data(t);
    uint k0 = uint(t.x) & 3u;
    uint k1 = (k0 + 1u) & 3u;
    uint k2 = (k0 + 2u) & 3u;
    uint k3 = (k0 + 3u) & 3u;
    cell->w[k0] = args.seed + 16u + k0;
    cell->w[k1] = args.seed + 16u + k1;
    cell->w[k2] = args.seed + 16u + k2;
    cell->w[k3] = args.seed + 16u + k3;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    uint4 w = uint4(cell->w[0], cell->w[1], cell->w[2], cell->w[3]);
    uint2 p = uint2(g) * uint2(n) + uint2(t);
    if (p.x < args.size.x && p.y < args.size.y) {
        uint i = p.y * args.size.x + p.x;
        out[4u * i] = w.x;
        out[4u * i + 1u] = w.y;
        out[4u * i + 2u] = w.z;
        out[4u * i + 3u] = w.w;
    }
}
