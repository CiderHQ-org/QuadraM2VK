// Owned fixture (written for this repository): a fragment function's
// simdgroup lane, read as a `uint` parameter, a `ushort` parameter and a Metal 3.1 program-scope
// `uint`, and its simdgroup width as a parameter, each written to an r32_uint target; and the lane
// beside the simdgroup operations that name lanes, written to an rgba32_uint target:
// the lane, a shuffle of the lane from itself, the first active lane by the ballot's bits, and a
// shuffle of the lane from that first lane. The vertex function places each vertex at the
// clip-space position buffer 0 holds, as given, so a draw's coverage is the positions' alone.
// Loop-free.
// metal-std: metal3.1
#include <metal_stdlib>
using namespace metal;

vertex float4 fragment_lanes_vertex(uint vid [[vertex_id]], const device float2* positions [[buffer(0)]])
{
    return float4(positions[vid], 0.0, 1.0);
}

fragment uint fragment_lane(uint lane [[thread_index_in_simdgroup]])
{
    return lane;
}

fragment uint fragment_lane_ushort(ushort lane [[thread_index_in_simdgroup]])
{
    return lane;
}

uint simd_lane [[thread_index_in_simdgroup]];

fragment uint fragment_global_lane()
{
    return simd_lane;
}

fragment uint fragment_simd_width(uint width [[threads_per_simdgroup]])
{
    return width;
}

fragment uint4 fragment_lane_shuffles(uint lane [[thread_index_in_simdgroup]])
{
    uint first = ctz(uint(static_cast<simd_vote::vote_t>(simd_ballot(true))));
    return uint4(lane, simd_shuffle(lane, ushort(lane)), first, simd_shuffle(lane, ushort(first)));
}
