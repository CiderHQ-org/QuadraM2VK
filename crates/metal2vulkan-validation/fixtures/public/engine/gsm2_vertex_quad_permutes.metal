// Owned fixture (written for this repository): quad broadcasts and
// quad shuffle-xors by a constant lane or mask in vertex functions, which the translation emits as
// Vulkan's quad instructions where the stage takes them and as shuffles elsewhere (MoltenVK's
// vertex stage). Rasterization-disabled points; vertex v reads in[v], a float whose bits differ
// from every other vertex's.
// - quad_permutes_in_a_vertex_function writes out[12v .. 12v + 11]: the broadcasts of each quad
//   lane 0 to 3 and the shuffle-xors by masks 1, 2 and 3 of in[v]; a float2's broadcast of lane 3;
//   a half's shuffle-xor by 2; an int's broadcast of lane 1; a short's shuffle-xor by 3; a uchar's
//   broadcast of lane 2. Every quad it runs over must be whole: a read of a lane the quad lacks is
//   undefined in Metal.
// - quad_pair_permutes_in_a_vertex_function writes out[4v .. 4v + 3]: the broadcasts of quad lanes
//   0 and 1, the shuffle-xor by 1, and an int's broadcast of lane 0, which read only lanes a quad
//   of two vertices has.
// - quad_permutes_in_branches_of_a_vertex_function writes out[6v .. 6v + 5] in two branches: where
//   v & 2 is 0 (quad lanes 0 and 1) the broadcasts of lanes 1 and 0 and the shuffle-xor by 1; where
//   v & 1 is 0 (quad lanes 0 and 2) the broadcasts of lanes 2 and 0 and the shuffle-xor by 2. Every
//   lane read is active; the other words keep the buffer's initial bytes.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

vertex void quad_permutes_in_a_vertex_function(uint vid [[vertex_id]],
                                               const device float *in [[buffer(0)]],
                                               device uint *out [[buffer(1)]])
{
    const float x = in[vid];
    out[12 * vid + 0] = as_type<uint>(quad_broadcast(x, ushort(0)));
    out[12 * vid + 1] = as_type<uint>(quad_broadcast(x, ushort(1)));
    out[12 * vid + 2] = as_type<uint>(quad_broadcast(x, ushort(2)));
    out[12 * vid + 3] = as_type<uint>(quad_broadcast(x, ushort(3)));
    out[12 * vid + 4] = as_type<uint>(quad_shuffle_xor(x, ushort(1)));
    out[12 * vid + 5] = as_type<uint>(quad_shuffle_xor(x, ushort(2)));
    out[12 * vid + 6] = as_type<uint>(quad_shuffle_xor(x, ushort(3)));
    out[12 * vid + 7] = as_type<uint>(quad_broadcast(float2(x, -x), ushort(3)).y);
    out[12 * vid + 8] = uint(as_type<ushort>(quad_shuffle_xor(half(x), ushort(2))));
    out[12 * vid + 9] = as_type<uint>(quad_broadcast(int(vid) - 20, ushort(1)));
    const short s = short(int(vid * 3u) - 50);
    out[12 * vid + 10] = as_type<uint>(int(quad_shuffle_xor(s, ushort(3))));
    out[12 * vid + 11] = uint(quad_broadcast(uchar(vid + 200u), ushort(2)));
}

vertex void quad_pair_permutes_in_a_vertex_function(uint vid [[vertex_id]],
                                                    const device float *in [[buffer(0)]],
                                                    device uint *out [[buffer(1)]])
{
    const float x = in[vid];
    out[4 * vid + 0] = as_type<uint>(quad_broadcast(x, ushort(0)));
    out[4 * vid + 1] = as_type<uint>(quad_broadcast(x, ushort(1)));
    out[4 * vid + 2] = as_type<uint>(quad_shuffle_xor(x, ushort(1)));
    out[4 * vid + 3] = as_type<uint>(quad_broadcast(int(vid) - 20, ushort(0)));
}

vertex void quad_permutes_in_branches_of_a_vertex_function(uint vid [[vertex_id]],
                                                           const device float *in [[buffer(0)]],
                                                           device uint *out [[buffer(1)]])
{
    const float x = in[vid];
    if ((vid & 2u) == 0u) {
        out[6 * vid + 0] = as_type<uint>(quad_broadcast(x, ushort(1)));
        out[6 * vid + 1] = as_type<uint>(quad_broadcast(x, ushort(0)));
        out[6 * vid + 2] = as_type<uint>(quad_shuffle_xor(x, ushort(1)));
    }
    if ((vid & 1u) == 0u) {
        out[6 * vid + 3] = as_type<uint>(quad_broadcast(x, ushort(2)));
        out[6 * vid + 4] = as_type<uint>(quad_broadcast(x, ushort(0)));
        out[6 * vid + 5] = as_type<uint>(quad_shuffle_xor(x, ushort(2)));
    }
}
