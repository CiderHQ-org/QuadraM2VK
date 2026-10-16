// Owned fixture: reads of a depth multisampled texture array
// (`air.read_depth_2d_ms_array`) at 32- and 16-bit coordinates. Lane i reads sample i mod s of
// texel ((i / s) mod w, (i / sw) mod h) in slice (i / swh) mod n and writes the two depths' words
// to out[2i .. 2i + 1]. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void unk_read_depth_ms_array(depth2d_ms_array<float, access::read> t [[texture(0)]],
                                    device uint2 *out [[buffer(0)]],
                                    uint i [[thread_position_in_grid]]) {
    uint w = t.get_width();
    uint h = t.get_height();
    uint s = t.get_num_samples();
    uint sample = i % s;
    uint2 xy = uint2((i / s) % w, (i / (s * w)) % h);
    uint slice = (i / (s * w * h)) % t.get_array_size();
    out[i] = as_type<uint2>(float2(t.read(xy, slice, sample),
                                   t.read(ushort2(xy), ushort(slice), ushort(sample))));
}
