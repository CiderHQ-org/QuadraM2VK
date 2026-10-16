// Owned fixture: texture atomics in and past read_write 2D textures, the
// probe of what Metal's `atomic_fetch_max` returns out of bounds (MSL 4.1 states only that an
// out-of-bounds write is ignored and an out-of-bounds read returns zero). Eight threads; thread t
// maxes 0xA0000000 | t and then 0x90000000 | t into the uint texture u at texel idx[t] (x in the
// low 16 bits, y in the high), keeping both returned old texels in out[3t] and out[3t + 1], and
// maxes 0x10000 + t into the int texture s at texel idx[8 + t], keeping the returned old texel in
// out[3t + 2]. The case's idx puts threads 0 to 3 on distinct texels inside each texture and
// threads 4 to 7 past them. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void trh_texture_atomic(device const uint *idx [[buffer(0)]],
                               texture2d<uint, access::read_write> u [[texture(0)]],
                               texture2d<int, access::read_write> s [[texture(1)]],
                               device uint *out [[buffer(1)]],
                               uint t [[thread_position_in_grid]]) {
    uint c = idx[t];
    uint2 xy = uint2(c & 0xFFFFu, c >> 16);
    out[3 * t] = u.atomic_fetch_max(xy, uint4(0xA0000000u | t)).x;
    out[3 * t + 1] = u.atomic_fetch_max(xy, uint4(0x90000000u | t)).x;
    uint d = idx[8 + t];
    out[3 * t + 2] = as_type<uint>(s.atomic_fetch_max(uint2(d & 0xFFFFu, d >> 16), int4(0x10000 + int(t))).x);
}
