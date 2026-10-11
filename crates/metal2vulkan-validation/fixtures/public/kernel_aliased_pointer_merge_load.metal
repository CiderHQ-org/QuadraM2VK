#include <metal_stdlib>
using namespace metal;

// A pointer merged from two DISTINCT device buffers and then aliased by the identity pointer cast
// the frontend spells `bitcast ptr %sel to ptr`, with the index applied to the ALIAS. Logical
// SPIR-V cannot select between two descriptors, so the merge lives in a side table with no pointer
// id of its own; the alias has to carry the merge or there is nothing for the index to land on.
//
// The condition is per-lane, so both arms are read in one dispatch: lanes 0 and 2 take `a`, lanes
// 1 and 3 take `b`.
kernel void load_through_an_aliased_pointer_merge(
    device const uchar *a [[buffer(0)]],
    device const uchar *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &pick [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  device const float *pa = reinterpret_cast<device const float *>(a + 8);
  device const float *pb = reinterpret_cast<device const float *>(b + 4);
  device const float *p = (pick & (1u << gid)) != 0u ? pa : pb;
  out[gid] = p[gid];
}
