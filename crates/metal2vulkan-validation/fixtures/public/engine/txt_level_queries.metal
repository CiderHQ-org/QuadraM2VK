// Owned fixture: the level count of storage textures, which a storage image
// cannot query, and a storage texture's size at a level read from a buffer. The binder pushes the
// bound texture's level count; the size query reads the level's view. out = (levels of t, width
// and height of t at level `lod`, levels of the cube c). A case passes level 0. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void txt_level_queries(texture2d<float, access::write> t [[texture(0)]],
                              texturecube<half, access::write> c [[texture(1)]],
                              constant uint &lod [[buffer(0)]],
                              device uint4 *out [[buffer(1)]]) {
    out[0] = uint4(t.get_num_mip_levels(), t.get_width(lod), t.get_height(lod),
                   c.get_num_mip_levels());
}
