// Owned fixture: the size queries the decoder once lacked -- the
// width and height of a multisampled texture array, of a depth multisampled texture and texture
// array, and of a depth cube and cube array (at level 0), and the width of a texture buffer --
// beside the array sizes and sample counts of the same textures. One lane writes them all.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void unk_size_queries(texture2d_ms_array<float, access::read> msa [[texture(0)]],
                             texture_buffer<float, access::read> buf [[texture(1)]],
                             depth2d_ms<float, access::read> dms [[texture(2)]],
                             depth2d_ms_array<float, access::read> dmsa [[texture(3)]],
                             depthcube<float> dc [[texture(4)]],
                             depthcube_array<float> dca [[texture(5)]],
                             device uint4 *out [[buffer(0)]]) {
    out[0] = uint4(msa.get_width(), msa.get_height(), msa.get_array_size(),
                   msa.get_num_samples());
    out[1] = uint4(buf.get_width(), dms.get_width(), dms.get_height(), dms.get_num_samples());
    out[2] = uint4(dmsa.get_width(), dmsa.get_height(), dmsa.get_array_size(),
                   dmsa.get_num_samples());
    out[3] = uint4(dc.get_width(), dc.get_height(), dca.get_width(), dca.get_height());
    out[4] = uint4(dca.get_array_size(), dc.get_num_mip_levels(), dca.get_num_mip_levels(), 0);
}
