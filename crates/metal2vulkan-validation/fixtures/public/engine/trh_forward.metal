// Owned fixture: a thread writes a texel of a read_write 2D texture and of a
// read_write texture buffer and then reads a texel of each, where a thread writing out of bounds
// reads back the location it wrote. Metal ignores the out-of-bounds write and reads zero there
// (MSL 4.1 sections 2.9.1 and 6.13; measured); Vulkan's robustness may return the value
// the thread stored unless the load is volatile. Eight threads; thread t writes 0xB0000000 | t to
// img at texel w = idx[t] (x in the low 16 bits, y in the high), reads img at texel idx[8 + t]
// into out[2t], writes 0xC0000000 | t to tb at element idx[16 + t], and reads tb at element
// idx[24 + t] into out[2t + 1]. The case's idx makes threads 0 to 3 write and read one location
// past each resource, and threads 4 to 7 write texels no thread reads and read texels no thread
// writes. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void trh_forward(device const uint *idx [[buffer(0)]],
                        texture2d<uint, access::read_write> img [[texture(0)]],
                        texture_buffer<uint, access::read_write> tb [[texture(1)]],
                        device uint *out [[buffer(1)]],
                        uint t [[thread_position_in_grid]]) {
    uint w = idx[t];
    uint r = idx[8 + t];
    img.write(uint4(0xB0000000u | t), uint2(w & 0xFFFFu, w >> 16));
    out[2 * t] = img.read(uint2(r & 0xFFFFu, r >> 16)).x;
    tb.write(uint4(0xC0000000u | t), idx[16 + t]);
    out[2 * t + 1] = tb.read(idx[24 + t]).x;
}
