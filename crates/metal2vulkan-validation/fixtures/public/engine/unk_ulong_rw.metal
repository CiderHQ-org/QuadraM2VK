// Owned fixture: reads and writes of Metal's `ulong` textures,
// which Metal binds only to RG32Uint textures (the low word in R, the high word in G).
//
// unk_ulong_rw: thread t reads one texel of each of a read_write 1D (8 texels), 1D array (8 x 2
// layers), 2D (8 x 2), 2D array (8 x 2 x 2 layers), 3D (8 x 2 x 2), and buffer (8 texels) ulong
// texture, at x = t for t < 10 and x = t & 7 after, row, slice, and layer t & 1 (the 3D's z
// t >> 1 & 1) for t < 10 and 2 after: threads 8 and 9 are past every texture's width, threads 10
// and 11 past the 2D and 3D textures' height, the arrays' last layer, and the 3D's depth (the 1D
// and buffer textures' x stays past their width). It writes each read's four lanes to
// out[6t .. 6t + 5], then writes back, at the same texel, a ulong4 whose lanes differ, lane 0
// being v ^ 0xF00D0000C0DE0000 + t. The 2D texture is read and written at 16-bit coordinates.
//
// unk_ulong_access: thread t reads texel (t & 7, t >> 3 & 1) of a sample-access 2D ulong texture
// (the read sampler's form) and of a read-access one (16-bit coordinates), writes both to out[2t]
// and out[2t + 1], and writes their sum to texel (t & 7, t >> 3 & 1, 0) of a write-access 3D
// ulong texture; threads 16 and on are past every texture.
//
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void unk_ulong_rw(texture1d<ulong, access::read_write> a1 [[texture(0)]],
                         texture1d_array<ulong, access::read_write> a1a [[texture(1)]],
                         texture2d<ulong, access::read_write> a2 [[texture(2)]],
                         texture2d_array<ulong, access::read_write> a2a [[texture(3)]],
                         texture3d<ulong, access::read_write> a3 [[texture(4)]],
                         texture_buffer<ulong, access::read_write> ab [[texture(5)]],
                         device ulong4 *out [[buffer(0)]],
                         uint t [[thread_position_in_grid]]) {
    uint x = t < 10 ? t : (t & 7);
    uint y = t < 10 ? (t & 1) : 2;
    uint z = t < 10 ? ((t >> 1) & 1) : 2;
    device ulong4 *o = out + 6 * t;
    ulong4 r1 = a1.read(t);
    ulong4 r1a = a1a.read(x, y);
    ulong4 r2 = a2.read(ushort2(x, y));
    ulong4 r2a = a2a.read(uint2(x, y), y);
    ulong4 r3 = a3.read(uint3(x, y, z));
    ulong4 rb = ab.read(t);
    o[0] = r1;
    o[1] = r1a;
    o[2] = r2;
    o[3] = r2a;
    o[4] = r3;
    o[5] = rb;
    ulong k = 0xF00D0000C0DE0000ul + t;
    a1.write(ulong4(r1.x ^ k, 1, 2, 3), t);
    a1a.write(ulong4(r1a.x ^ k, 4, 5, 6), x, y);
    a2.write(ulong4(r2.x ^ k, 7, 8, 9), ushort2(x, y));
    a2a.write(ulong4(r2a.x ^ k, 10, 11, 12), uint2(x, y), y);
    a3.write(ulong4(r3.x ^ k, 13, 14, 15), uint3(x, y, z));
    ab.write(ulong4(rb.x ^ k, 16, 17, 18), t);
}

kernel void unk_ulong_access(texture2d<ulong> s [[texture(0)]],
                             texture2d<ulong, access::read> r [[texture(1)]],
                             texture3d<ulong, access::write> w [[texture(2)]],
                             device ulong4 *out [[buffer(0)]],
                             uint t [[thread_position_in_grid]]) {
    uint2 c = uint2(t & 7, (t >> 3) & 1);
    if (t >= 16) {
        c += uint2(0, 2);
    }
    ulong4 sv = s.read(c);
    ulong4 rv = r.read(ushort2(c));
    out[2 * t] = sv;
    out[2 * t + 1] = rv;
    w.write(ulong4(sv.x + rv.x, 0, 0, 0), uint3(c, 0));
}
