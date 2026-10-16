// Owned fixture: reads of a depth cube and a depth cube array
// (`air.read_depth_cube`, `air.read_depth_cube_array`) at 32- and 16-bit coordinates. Lane i
// reads face i mod 6 at texel ((i / 6) mod w, (i / 6w) mod h) of a w x h depth cube, and of cube
// (i / 6wh) mod n of a depth cube array of n cubes, and writes the four depths' words to
// out[4i .. 4i + 3]. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void unk_read_depth_cube(depthcube<float, access::read> c [[texture(0)]],
                                depthcube_array<float, access::read> a [[texture(1)]],
                                device uint4 *out [[buffer(0)]],
                                uint i [[thread_position_in_grid]]) {
    uint w = c.get_width();
    uint h = c.get_height();
    uint face = i % 6;
    uint2 xy = uint2((i / 6) % w, (i / (6 * w)) % h);
    uint cube = (i / (6 * w * h)) % a.get_array_size();
    out[i] = as_type<uint4>(float4(c.read(xy, face), c.read(ushort2(xy), ushort(face)),
                                   a.read(xy, face, cube),
                                   a.read(ushort2(xy), ushort(face), ushort(cube))));
}
