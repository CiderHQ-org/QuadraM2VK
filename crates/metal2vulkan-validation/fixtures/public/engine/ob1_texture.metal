// Owned fixture: reads and writes past a 2D texture, the texture
// containment form, whose results Metal defines (MSL 4.1: a read out of bounds returns zero, a
// write out of bounds is ignored). Eight threads; thread t reads src at the texel idx[t] names (x in
// the low 16 bits, y in the high) into out[t] and writes 0xA0000000 | t to dst at the texel
// idx[8 + t] names. The case's idx puts reads and writes 0 to 3 on the four texels of the 2x2
// textures and reads and writes 4 to 7 past them. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void ob1_texture(device const uint *idx [[buffer(0)]],
                        texture2d<uint, access::read> src [[texture(0)]],
                        texture2d<uint, access::write> dst [[texture(1)]],
                        device uint *out [[buffer(1)]],
                        uint t [[thread_position_in_grid]]) {
    out[t] = src.read(uint2(idx[t] & 0xFFFFu, idx[t] >> 16)).x;
    dst.write(uint4(0xA0000000u | t), uint2(idx[8 + t] & 0xFFFFu, idx[8 + t] >> 16));
}
