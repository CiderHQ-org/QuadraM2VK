// Owned fixture: size queries of a sampled texture at a level read from a
// buffer (`texture.query_clamped`: under containment a level that is not the constant 0 is
// clamped to the last level). A case passes level 0, which names a level of every texture:
// out = (width, height, levels). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_query_dynamic_level(texture2d<float> t [[texture(0)]],
                                   constant uint &lod [[buffer(0)]],
                                   device uint *out [[buffer(1)]]) {
    out[0] = t.get_width(lod);
    out[1] = t.get_height(lod);
    out[2] = t.get_num_mip_levels();
}
