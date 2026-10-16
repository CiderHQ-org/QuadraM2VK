// Owned fixture: textures of an argument buffer (its heap): a read, a
// size query, a null test, and a write (`access.handle.lookup`, `texture.read.heap`,
// `texture.query.heap`, `texture.is_null.heap`, `texture.write.heap`). The argument buffer holds a
// sampled texture and a storage texture. Thread g writes texel g of the sampled texture plus
// (its width, whether it is null, 0, 0) at texel g of the storage texture. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Resources {
    texture2d<float> source;
    texture2d<float, access::write> target;
};

kernel void a4_heap_textures(constant Resources &res [[buffer(0)]],
                             uint gid [[thread_position_in_grid]]) {
    float4 texel = res.source.read(uint2(gid, 0));
    float none = is_null_texture(res.source) ? 1.0 : 0.0;
    res.target.write(texel + float4(float(res.source.get_width()), none, 0.0, 0.0),
                     uint2(gid, 0));
}
