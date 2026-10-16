// Owned fixture: tile functions over their tile's threadgroup memory,
// which a render pass keeps across its tile dispatches. Metal gives each tile one allocation of
// the pass's `threadgroupMemoryLength` bytes, zero when the pass begins, and places each
// `[[threadgroup(n)]]` argument at the byte offset `setThreadgroupMemoryLength:offset:atIndex:`
// gives it, so arguments at overlapping offsets share words (measured on an M4 and an M3 Max).
// Each runs in a tile pipeline of one `r32uint` color attachment it does not touch. The tile's
// index is `x + y * width` of its `threadgroup_position_in_grid` over `threadgroups_per_grid`.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

static uint tile_of(uint2 g, uint2 n)
{
    return g.y * n.x + g.x;
}

// Word t of memory 0 grows by 1000 * tile + t + 1, t the thread's index in the tile: k
// dispatches over a pass's tile leave k times that in the word.
kernel void cfit_tile_memory_add(threadgroup uint *tg [[threadgroup(0)]],
                                 uint t [[thread_index_in_threadgroup]],
                                 uint2 g [[threadgroup_position_in_grid]],
                                 uint2 n [[threadgroups_per_grid]])
{
    tg[t] += 1000u * tile_of(g, n) + t + 1u;
}

// out[tile * count + t] = word t of memory 0, count the threads of the tile.
kernel void cfit_tile_memory_dump(threadgroup uint *tg [[threadgroup(0)]],
                                  device uint *out [[buffer(0)]],
                                  uint t [[thread_index_in_threadgroup]],
                                  uint2 size [[threads_per_threadgroup]],
                                  uint2 g [[threadgroup_position_in_grid]],
                                  uint2 n [[threadgroups_per_grid]])
{
    out[tile_of(g, n) * size.x * size.y + t] = tg[t];
}

// Word t of memory 1 takes word t of memory 0 plus 0x10000, once every thread read its word of
// memory 0. Placed at overlapping offsets, memory 1's words are memory 0's further on.
kernel void cfit_tile_memory_shift(threadgroup uint *a [[threadgroup(0)]],
                                   threadgroup uint *b [[threadgroup(1)]],
                                   uint t [[thread_index_in_threadgroup]])
{
    uint v = a[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    b[t] = v + 0x10000u;
}
