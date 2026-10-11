#include <metal_stdlib>
using namespace metal;

// `Rec` has 17 bytes of members and an alignment of 2, so Metal's sizeof is 18 -- NOT a multiple of
// four. `sizes[tid]` reports `sizeof * 1000 + alignof` so the case records Metal's own answer, and
// the two stores land at 18*tid + 6 and 18*tid + 11. A translator that rounds the record up to 20
// writes them at 20*tid instead, which every thread but the first shows.
struct Rec {
    short a[5];
    uchar b[7];
};

kernel void record_stride_not_a_multiple_of_four(device Rec *recs [[buffer(0)]],
                                                 device uint *sizes [[buffer(1)]],
                                                 uint tid [[thread_position_in_grid]]) {
    sizes[tid] = (uint)sizeof(Rec) * 1000u + (uint)alignof(Rec);
    recs[tid].a[3] = (short)(tid * 7 + 1);
    recs[tid].b[1] = (uchar)(tid ^ 0xa5);
}
