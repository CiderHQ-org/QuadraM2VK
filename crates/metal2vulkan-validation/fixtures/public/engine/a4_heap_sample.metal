// Owned fixture: an argument-buffer texture sampled through constexpr
// samplers (`sampler.heap.sample`, `sampler.heap.pixel_fetch`): a normalized nearest sample
// and a nearest pixel sample (an exact fetch), each at the centre of texel (x, y) of a 4 x 2
// texture, so both read that texel: target(x, y) = 3 source(x, y). Loop-free.
#include <metal_stdlib>
using namespace metal;

constexpr sampler nearest_clamp(coord::normalized, filter::nearest, address::clamp_to_edge);
constexpr sampler pixel_nearest(coord::pixel, filter::nearest, address::clamp_to_edge);

struct Copy {
    texture2d<float> source;
    texture2d<float, access::write> target;
};

kernel void a4_heap_sample(constant Copy &copy [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    float2 uv = (float2(gid) + 0.5) * float2(0.25, 0.5);
    copy.target.write(copy.source.sample(nearest_clamp, uv)
                          + copy.source.sample(pixel_nearest, float2(gid) + 0.5) * 2.0,
                      gid);
}
