// Owned fixture: loads and stores past a threadgroup allocation, the
// threadgroup containment form. One threadgroup of eight threads over a 16-word allocation; thread
// t stores 0xB0000000 | t to tg[t] and 0xC0000000 | t to tg[idx[t]], and after a barrier loads
// tg[idx[8 + t]] into out[t]. The case's idx puts stores 0 to 3 in tg (words 8 to 11) and stores 4
// to 7 past its end (words 16, 20, 33, 79: up to byte 320), loads 0 to 3 from written words (0, 3,
// 8, 11) and loads 4 to 7 past its end (words 16, 20, 40, 79). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void ob1_threadgroup(device const uint *idx [[buffer(0)]],
                            device uint *out [[buffer(1)]],
                            threadgroup uint *tg [[threadgroup(0)]],
                            uint t [[thread_position_in_grid]]) {
    tg[t] = 0xB0000000u | t;
    tg[idx[t]] = 0xC0000000u | t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[t] = tg[idx[8 + t]];
}
