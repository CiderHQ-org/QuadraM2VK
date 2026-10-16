// Owned fixture: which thread of a tile dispatch is which. Each thread
// adds one to its own cell of an `r32float` attachment and records, at its
// `thread_index_in_threadgroup`, its `thread_position_in_threadgroup`, its lane and simdgroup, and
// the position of its simdgroup's first active lane (`simd_broadcast_first`), so a case shows how
// a tile dispatch orders the threads of its threadgroup and which threads share a simdgroup.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Px {
    float f [[color(0)]];
};

kernel void k10ts_tile_order(
    imageblock<Px, imageblock_layout_implicit> block,
    device uint *out [[buffer(0)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort tidx [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    Px p = block.read(tid);
    block.write(Px{p.f + 1.0f}, tid);
    uint position = uint(tid.x) | (uint(tid.y) << 8);
    device uint *r = out + uint(tidx) * 2u;
    r[0] = position | (uint(lane) << 16) | (uint(sg) << 24);
    r[1] = simd_broadcast_first(position);
}
