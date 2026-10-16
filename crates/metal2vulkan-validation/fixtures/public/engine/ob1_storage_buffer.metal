// Owned fixture: loads and stores past a `device` buffer, the storage
// buffer containment form. Eight threads; thread t loads src[idx[t]] into out[t] and stores
// 0xA0000000 | t to out[idx[8 + t]]. The case's idx puts loads 0 to 3 in src (16 bytes) and loads
// 4 to 7 past its end (words 4, 5, 17, 63: up to byte 256), stores 0 to 3 in out (words 8 to 11 of
// 16) and stores 4 to 7 past its end (words 16, 17, 40, 79: up to byte 320). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void ob1_storage_buffer(device const uint *idx [[buffer(0)]],
                               device const uint *src [[buffer(1)]],
                               device uint *out [[buffer(2)]],
                               uint t [[thread_position_in_grid]]) {
    out[t] = src[idx[t]];
    out[idx[8 + t]] = 0xA0000000u | t;
}
