// A kernel whose three writes are the corpus's uncovered *short-coordinate* texture writes:
// `air.write_texture_2d_array.i16.u.v4i32`, `air.write_texture_2d_array.i16.u.v4i16` and
// `air.write_texture_2d.i16.s.v4i16`. The coordinate is a `ushort2` (and the array slice a
// `ushort`), which is a different AIR operand shape from the `uint2` spellings the corpus already
// covers -- the coordinate word is 16 bits wide, so a lowering that sign-extends it, or that
// truncates the slice, addresses a different texel than Metal does.
//
// Every destination texel gets a distinct value derived from its own (x, y, slice), so a swapped
// coordinate pair, a dropped slice or a wrong lane order all show up as a content mismatch rather
// than as a silently identical image. The values themselves are chosen outside the next narrower
// range: the uint4 lanes exceed 16 bits, the ushort4 lanes exceed 8 bits, and the short4 lanes lie
// outside the signed byte range, so a storage image declared one step too narrow cannot carry them.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_write_short_coordinate_textures(
    texture2d_array<uint, access::write> wide [[texture(0)]],
    texture2d_array<ushort, access::write> narrow [[texture(1)]],
    texture2d<short, access::write> signed_plane [[texture(2)]],
    ushort2 gid [[thread_position_in_grid]])
{
    for (ushort slice = 0; slice < 3; ++slice) {
        ushort idx = ushort(slice * 16 + gid.y * 4 + gid.x);
        wide.write(uint4(0xDEADBEEFu ^ uint(idx),
                         0x80000000u + uint(idx),
                         0x00010000u * uint(gid.x + 1),
                         0x7FFFFFFFu - uint(idx)),
                   gid, slice);
        narrow.write(ushort4(0xFFFFu - idx,
                             0x8000u + idx,
                             ushort(256 + gid.x),
                             ushort(300 + gid.y)),
                     gid, slice);
    }
    ushort plane = ushort(gid.y * 4 + gid.x);
    signed_plane.write(short4(short(-32768 + plane),
                              short(-129 - gid.x),
                              short(128 + gid.y),
                              short(32767 - plane)),
                       gid);
}
