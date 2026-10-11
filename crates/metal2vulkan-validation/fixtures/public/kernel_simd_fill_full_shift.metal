// `simd_shuffle_and_fill_{up,down}` asked to shift a whole cluster out, which is the one input the
// two mirror lowerings disagree about.
//
// Both are defined the same way in opposite directions: within each `modulo`-wide cluster, lane `l`
// reads `data[l -+ delta]`, and a lane whose source leaves the cluster reads `filling_data` at the
// wrapped position instead. At `delta == modulo` every lane's source leaves the cluster, so every
// lane must answer `filling_data` at its own position -- and `delta == modulo` is not an exotic
// input: the two-argument overload passes `modulo = __metal_get_simdgroup_size()`, so
// `simd_shuffle_and_fill_up(data, fill, 32)` on a 32-wide simdgroup spells exactly this.
//
// `lower_simd_shuffle_and_fill_up` reduces `delta` modulo the cluster width before comparing it
// against the lane, so at `delta == modulo` it compares against zero, finds every lane in bounds,
// and answers `data` from the lane's own position. `lower_simd_shuffle_and_fill_down` does not
// reduce, finds every lane out of bounds, and answers `filling_data`. They cannot both be right.
//
// Eight columns per lane, and each one is derived rather than recorded. `data` is `lane + 100` and
// `filling_data` is `lane + 200`, so every one of the 64 values in play is distinct and a
// data-vs-fill confusion, a lane-index error and a cluster-base error all land on different
// numbers. Columns 0, 1, 6 and 7 are ordinary in-range deltas that pin the model itself -- if any of
// those disagrees, the disagreement is about the family and not about the full shift.
//
//     col 0  up   delta 0  modulo 32   data[l]                            = l + 100
//     col 1  up   delta 1  modulo 32   l == 0 ? fill[31] : data[l - 1]    = 231 or l + 99
//     col 2  up   delta 32 modulo 32   fill[l]                            = l + 200
//     col 3  down delta 32 modulo 32   fill[l]                            = l + 200
//     col 4  up   delta 8  modulo 8    fill[l]                            = l + 200
//     col 5  down delta 8  modulo 8    fill[l]                            = l + 200
//     col 6  up   delta 3  modulo 8    lc >= 3 ? data[b+lc-3] : fill[b+lc+5]
//     col 7  down delta 3  modulo 8    lc <  5 ? data[b+lc+3] : fill[b+lc-5]
//
// where `b = 8 * (l / 8)` is the cluster base and `lc = l % 8` the in-cluster lane. The dispatch is
// exactly 32 threads, one simdgroup, so no lane's shuffle source is outside the group and every
// column is defined at every lane.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_simd_fill_full_shift(
    device uint* out [[buffer(0)]],
    uint lane [[thread_position_in_grid]])
{
    uint data = lane + 100u;
    uint fill = lane + 200u;
    out[lane * 8 + 0] = simd_shuffle_and_fill_up(data, fill, ushort(0), ushort(32));
    out[lane * 8 + 1] = simd_shuffle_and_fill_up(data, fill, ushort(1), ushort(32));
    out[lane * 8 + 2] = simd_shuffle_and_fill_up(data, fill, ushort(32), ushort(32));
    out[lane * 8 + 3] = simd_shuffle_and_fill_down(data, fill, ushort(32), ushort(32));
    out[lane * 8 + 4] = simd_shuffle_and_fill_up(data, fill, ushort(8), ushort(8));
    out[lane * 8 + 5] = simd_shuffle_and_fill_down(data, fill, ushort(8), ushort(8));
    out[lane * 8 + 6] = simd_shuffle_and_fill_up(data, fill, ushort(3), ushort(8));
    out[lane * 8 + 7] = simd_shuffle_and_fill_down(data, fill, ushort(3), ushort(8));
}
