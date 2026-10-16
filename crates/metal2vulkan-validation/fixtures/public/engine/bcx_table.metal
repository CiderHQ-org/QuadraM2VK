// Owned fixture: the weights of Metal's bicubic filter (`filter::bicubic`)
// at every 1/256 fraction of each axis. One clamp-to-edge bicubic sampler with pixel coordinates
// samples `across` at each lane's coordinate and `down` at the same coordinate with its axes
// swapped; the cases bind `down` to the transpose of `across`, so one lane reads the same taps
// horizontally and vertically. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler bicubic_pixel(filter::bicubic, address::clamp_to_edge, coord::pixel);

kernel void bcx_table(texture2d<float, access::sample> across [[texture(0)]],
                      texture2d<float, access::sample> down [[texture(1)]],
                      const device float2 *coord [[buffer(0)]],
                      device float4 *out [[buffer(1)]],
                      uint tid [[thread_position_in_grid]])
{
    float2 p = coord[tid];
    out[2 * tid + 0] = across.sample(bicubic_pixel, p);
    out[2 * tid + 1] = down.sample(bicubic_pixel, p.yx);
}
