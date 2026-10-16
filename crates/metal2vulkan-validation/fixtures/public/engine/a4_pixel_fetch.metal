// Owned fixture: a nearest, clamp-to-edge pixel sampler at texel centres,
// an exact texel fetch (`sampler.pixel_fetch`): out[y * 4 + x] = t(x, y) over a 4 x 2 grid.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

constexpr sampler pixel_nearest(coord::pixel, filter::nearest, address::clamp_to_edge);

kernel void a4_pixel_fetch(texture2d<float> t [[texture(0)]],
                           device float4 *out [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * 4 + gid.x] = t.sample(pixel_nearest, float2(gid) + 0.5);
}
