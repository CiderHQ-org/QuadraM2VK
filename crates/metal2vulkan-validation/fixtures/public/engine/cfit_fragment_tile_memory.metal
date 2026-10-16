// Owned fixture: fragment functions over their tile's threadgroup memory,
// which a render pass keeps across its draws and tile dispatches. Metal gives each tile one
// allocation of the pass's `threadgroupMemoryLength` bytes, zero when the pass begins, shared by
// every fragment of the tile and by the pass's tile functions, each `[[threadgroup(n)]]` argument
// at the byte offset `setThreadgroupMemoryLength:offset:atIndex:` gives it (measured on an M4).
// The fragment functions write one `r32uint` color attachment; the tile functions run in a tile
// pipeline of that attachment and do not touch it. A tile function's tile index is `x + y *
// width` of its `threadgroup_position_in_grid` over `threadgroups_per_grid`, and a fragment's the
// same of its pixel's tile, `pass.x` tiles across of `pass.y` pixels square. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Corner {
    float4 position [[position]];
};

// One corner of the rectangle of pixels `rect.xy` to `rect.zw` (excluded) in a `size` render
// area, drawn as a strip of four vertices.
vertex Corner cfit_frag_tile_rect(uint vid [[vertex_id]],
                                  constant float4 &rect [[buffer(0)]],
                                  constant float2 &size [[buffer(1)]])
{
    float2 pixel = float2((vid & 1u) != 0u ? rect.z : rect.x, (vid & 2u) != 0u ? rect.w : rect.y);
    Corner out;
    out.position = float4(pixel.x / size.x * 2.0f - 1.0f, 1.0f - pixel.y / size.y * 2.0f, 0.0f, 1.0f);
    return out;
}

// Word 0 of memory 0 counts the fragments of the tile.
fragment uint cfit_frag_tile_count(threadgroup atomic_uint *tg [[threadgroup(0)]])
{
    atomic_fetch_add_explicit(&tg[0], 1u, memory_order_relaxed);
    return 1u;
}

// Word 1 of memory 0 takes `v`.
fragment uint cfit_frag_tile_store(threadgroup uint *tg [[threadgroup(0)]],
                                   constant uint &v [[buffer(0)]])
{
    tg[1] = v;
    return v;
}

// out[2 * tile] and out[2 * tile + 1] take words 0 and 1 of memory 0, and the pixel word 1.
fragment uint cfit_frag_tile_report(float4 position [[position]],
                                    threadgroup uint *tg [[threadgroup(0)]],
                                    device uint *out [[buffer(0)]],
                                    constant uint2 &pass [[buffer(1)]])
{
    uint2 pixel = uint2(position.xy);
    uint tile = (pixel.y / pass.y) * pass.x + pixel.x / pass.y;
    out[2u * tile] = tg[0];
    out[2u * tile + 1u] = tg[1];
    return tg[1];
}

// The splat rasterizer's shape: memory 0 is a bitset of the tile's pixels, bit `i % 32` of word
// `i / 32` the pixel `i = y * pass.y + x` inside the tile; each fragment sets its own.
fragment uint cfit_frag_tile_or(float4 position [[position]],
                                threadgroup atomic_uint *bits [[threadgroup(0)]],
                                constant uint2 &pass [[buffer(1)]])
{
    uint2 pixel = uint2(position.xy) % pass.y;
    uint i = pixel.y * pass.y + pixel.x;
    atomic_fetch_or_explicit(&bits[i / 32u], 1u << (i % 32u), memory_order_relaxed);
    return 2u;
}

static uint tile_of(uint2 g, uint2 n)
{
    return g.y * n.x + g.x;
}

// Word t of memory 0 takes 0, t the thread's index in the tile, for t below `words`.
kernel void cfit_tile_bits_clear(threadgroup uint *bits [[threadgroup(0)]],
                                 constant uint &words [[buffer(1)]],
                                 uint t [[thread_index_in_threadgroup]])
{
    if (t < words) {
        bits[t] = 0u;
    }
}

// out[tile * words + t] gains word t of memory 0, for t below `words`.
kernel void cfit_tile_bits_resolve(threadgroup uint *bits [[threadgroup(0)]],
                                   device atomic_uint *out [[buffer(0)]],
                                   constant uint &words [[buffer(1)]],
                                   uint t [[thread_index_in_threadgroup]],
                                   uint2 g [[threadgroup_position_in_grid]],
                                   uint2 n [[threadgroups_per_grid]])
{
    if (t < words) {
        atomic_fetch_or_explicit(&out[tile_of(g, n) * words + t], bits[t], memory_order_relaxed);
    }
}
