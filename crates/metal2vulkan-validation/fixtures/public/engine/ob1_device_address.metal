// Owned fixture: loads and stores past a `device` buffer whose address
// escapes (its pointer is converted to an integer), so the product reaches it by its device
// address (a wide pointer through its root entry), the device address containment form. Eight
// threads; thread t loads buf[idx[t]] into out[t], writes the low four bits of that element's
// address to out[8 + t], and stores 0xA0000000 | t to buf[idx[8 + t]]. The case's idx puts loads 0
// to 3 in buf (16 bytes) and loads 4 to 7 past its end (words 4, 5, 17, 63: up to byte 256), and
// every store past its end (words 4 to 63). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void ob1_device_address(device const uint *idx [[buffer(0)]],
                               device uint *buf [[buffer(1)]],
                               device uint *out [[buffer(2)]],
                               uint t [[thread_position_in_grid]]) {
    device uint *p = buf + idx[t];
    out[t] = *p;
    out[8 + t] = uint(reinterpret_cast<uintptr_t>(p)) & 0xFu;
    buf[idx[8 + t]] = 0xA0000000u | t;
}
