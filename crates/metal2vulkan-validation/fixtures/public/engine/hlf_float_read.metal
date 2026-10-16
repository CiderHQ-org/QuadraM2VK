// Owned fixture: the `float` results of the texels whose `half` results
// `bcx_half_read` reads, to tell Metal's decode and filter from its conversion to `half`. Four 2D
// `float` textures are each read at the lane's coordinate truncated to a texel, sampled through a
// nearest-filter sampler, and sampled through a linear-filter sampler, both clamp-to-edge with
// pixel coordinates. The cases bind a different format to each slot. Lane `i` writes its twelve
// results to `out[12i .. 12i + 11]` in slot order (read, nearest, linear per slot). Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_pixel(filter::nearest, address::clamp_to_edge, coord::pixel);
constant sampler linear_pixel(filter::linear, address::clamp_to_edge, coord::pixel);

kernel void hlf_float_read(texture2d<float, access::sample> t0 [[texture(0)]],
                           texture2d<float, access::sample> t1 [[texture(1)]],
                           texture2d<float, access::sample> t2 [[texture(2)]],
                           texture2d<float, access::sample> t3 [[texture(3)]],
                           const device float2 *coord [[buffer(0)]],
                           device float4 *out [[buffer(1)]],
                           uint tid [[thread_position_in_grid]])
{
    float2 p = coord[tid];
    uint2 texel = uint2(p);
    device float4 *o = out + 12 * tid;
    o[0] = t0.read(texel);
    o[1] = t0.sample(nearest_pixel, p);
    o[2] = t0.sample(linear_pixel, p);
    o[3] = t1.read(texel);
    o[4] = t1.sample(nearest_pixel, p);
    o[5] = t1.sample(linear_pixel, p);
    o[6] = t2.read(texel);
    o[7] = t2.sample(nearest_pixel, p);
    o[8] = t2.sample(linear_pixel, p);
    o[9] = t3.read(texel);
    o[10] = t3.sample(nearest_pixel, p);
    o[11] = t3.sample(linear_pixel, p);
}
