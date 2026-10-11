// Six uncovered texture families that need no sampler: `air.read_texture_cube.v4f32` (4 corpus
// sources), `air.read_depth_2d_array.f32` (4), `air.read_texture_2d_array.i16.u.v4i16` (4),
// `air.get_width_texture_1d` (5), `air.is_null_texture_1d` (4) and `air.is_null_texture_3d` (4).
//
// Each of the four threads reads a different face, slice and texel, and every texel of every
// texture holds a distinct value, so a read that resolves the face, the array slice or the
// coordinate pair wrongly lands on a value that belongs to some other thread. The three reads also
// span three coordinate spellings on purpose: the cube and the depth array take `uint2` while the
// unsigned array takes `ushort2`, which is the `.i16` in its symbol.
//
// The two `is_null_texture` queries are the discriminating pair: `line` IS bound, so it must answer
// false and its width query must return the real width, while `vol` is deliberately left unbound,
// so it must answer true. A lowering that folds the query to a constant cannot satisfy both.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_texture_reads_and_queries(
    texturecube<float, access::read> cube [[texture(0)]],
    depth2d_array<float, access::read> darr [[texture(1)]],
    texture2d_array<ushort, access::read> uarr [[texture(2)]],
    texture1d<float, access::read> line [[texture(3)]],
    texture3d<float, access::read> vol [[texture(4)]],
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    uint x = tid % 2;
    uint y = tid / 2;
    float4 c = cube.read(uint2(x, y), tid);
    float d = darr.read(uint2(x, y), x);
    ushort4 u = uarr.read(ushort2(x, y), y);
    uint w = line.get_width();
    bool n1 = is_null_texture(line);
    bool n3 = is_null_texture(vol);

    uint b = tid * 7;
    out[b + 0] = as_type<uint>(c.x);
    out[b + 1] = as_type<uint>(c.w);
    out[b + 2] = as_type<uint>(d);
    out[b + 3] = uint(u.x);
    out[b + 4] = uint(u.w);
    out[b + 5] = w;
    out[b + 6] = (n1 ? 2u : 0u) + (n3 ? 1u : 0u);
}
