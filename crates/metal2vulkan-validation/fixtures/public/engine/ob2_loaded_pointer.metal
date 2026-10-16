// Owned fixture: loads and stores past a buffer reached through a pointer
// loaded from memory, the loaded pointer containment form. The argument buffer `h` holds the
// address of a separately owned 16-byte buffer, so `h.p` is a pointer the kernel loads from
// device memory (a dynamic root). Eight threads; thread t loads p[idx[t]] into out[t] and stores
// 0xB0000000 | t to p[idx[8 + t]]. The case's idx puts loads 0 to 3 in p's four words and loads 4
// to 7 past its end, and every store past its end. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Holder {
    device uint *p;
};

kernel void ob2_loaded_pointer(device const uint *idx [[buffer(0)]],
                               device const Holder &h [[buffer(1)]],
                               device uint *out [[buffer(2)]],
                               uint t [[thread_position_in_grid]]) {
    device uint *p = h.p;
    out[t] = p[idx[t]];
    p[idx[8 + t]] = 0xB0000000u | t;
}
