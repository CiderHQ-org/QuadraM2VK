// Owned fixture: Metal's atomic_min and atomic_max on ulong
// textures, which compare a texel's unsigned 64-bit value (the low word in R and the high word in
// G of the RG32Uint texture Metal binds) and return nothing.
//
// unk_ulong_atomics: thread t updates texel x = t & 15 of a read_write 1D, 1D array (layer
// t >> 4 & 1), 2D (row t >> 4 & 1, 16-bit coordinates), 2D array (row and layer t >> 4 & 1),
// 3D (row and slice t >> 4 & 1), and buffer ulong texture with v = t * 0x9E3779B97F4A7C15, whose
// top bit varies: atomic_max where x < 8, atomic_min elsewhere, so each texel takes one operation
// (the threads that share a texel commute). Threads 64 and on are past every texture's width.
//
// unk_ulong_atomic_rw: thread t reads texel (t, 0) of a read_write 2D ulong texture `a` that it
// also updates atomically (threads 8 and on read past its width), writes the read's four lanes to
// out[2t], updates texel (t & 7, 1) with atomic_max of the read's lane 0 xor k, and writes
// ulong4(r.x + 1, 2, 3, 4) back to texel (t, 0). It also reads texel (t, 0) of a read_write 2D
// ulong texture `b` that it never updates atomically, writes the read to out[2t + 1], and writes
// ulong4(r.x ^ k, 5, 6, 7) back there.
//
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void unk_ulong_atomics(texture1d<ulong, access::read_write> a1 [[texture(0)]],
                              texture1d_array<ulong, access::read_write> a1a [[texture(1)]],
                              texture2d<ulong, access::read_write> a2 [[texture(2)]],
                              texture2d_array<ulong, access::read_write> a2a [[texture(3)]],
                              texture3d<ulong, access::read_write> a3 [[texture(4)]],
                              texture_buffer<ulong, access::read_write> ab [[texture(5)]],
                              uint t [[thread_position_in_grid]]) {
    uint x = t & 15;
    uint y = (t >> 4) & 1;
    if (t >= 64) {
        x += 16;
    }
    ulong4 v = ulong4(ulong(t) * 0x9E3779B97F4A7C15ul, 0, 0, 0);
    if (x < 8) {
        a1.atomic_max(x, v);
        a1a.atomic_max(x, y, v);
        a2.atomic_max(ushort2(x, y), v);
        a2a.atomic_max(uint2(x, y), y, v);
        a3.atomic_max(uint3(x, y, y), v);
        ab.atomic_max(x, v);
    } else {
        a1.atomic_min(x, v);
        a1a.atomic_min(x, y, v);
        a2.atomic_min(ushort2(x, y), v);
        a2a.atomic_min(uint2(x, y), y, v);
        a3.atomic_min(uint3(x, y, y), v);
        ab.atomic_min(x, v);
    }
}

kernel void unk_ulong_atomic_rw(texture2d<ulong, access::read_write> a [[texture(0)]],
                                texture2d<ulong, access::read_write> b [[texture(1)]],
                                device ulong4 *out [[buffer(0)]],
                                uint t [[thread_position_in_grid]]) {
    ulong k = 0xF00D0000C0DE0000ul + t;
    ulong4 r = a.read(uint2(t, 0));
    out[2 * t] = r;
    a.atomic_max(uint2(t & 7, 1), ulong4(r.x ^ k, 0, 0, 0));
    a.write(ulong4(r.x + 1, 2, 3, 4), uint2(t, 0));
    ulong4 s = b.read(uint2(t, 0));
    out[2 * t + 1] = s;
    b.write(ulong4(s.x ^ k, 5, 6, 7), uint2(t, 0));
}
