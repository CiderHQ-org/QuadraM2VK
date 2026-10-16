// Owned fixture: reads and writes past a texture buffer, the texel buffer
// containment form, whose results Metal defines (MSL 4.1: a read out of bounds returns zero, a
// write out of bounds is ignored). Eight threads; thread t reads texel idx[t] of src into out[t]
// and writes 0xA0000000 | t to texel idx[8 + t] of dst. The case's idx puts reads and writes 0 to
// 3 on the four texels of each texture buffer and reads and writes 4 to 7 past them. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void ob1_texel_buffer(device const uint *idx [[buffer(0)]],
                             texture_buffer<uint, access::read> src [[texture(0)]],
                             texture_buffer<uint, access::write> dst [[texture(1)]],
                             device uint *out [[buffer(1)]],
                             uint t [[thread_position_in_grid]]) {
    out[t] = src.read(idx[t]).x;
    dst.write(uint4(0xA0000000u | t), idx[8 + t]);
}
