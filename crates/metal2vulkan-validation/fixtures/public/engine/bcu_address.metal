// Owned fixture: Metal's bicubic filter at the texture's edges under each
// address mode, and with pixel coordinates. Each lane reads a normalized coordinate `(u, v)` and
// the same point in pixels `(x, y)` (both exact, from the input buffer) and samples one texture
// through bicubic samplers that clamp to the edge, clamp to zero, clamp to an opaque white border,
// repeat, and mirror-repeat (normalized), and that clamp to the edge and to zero with pixel
// coordinates. A footprint that crosses an edge shows how each of its taps is addressed.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler bicubic_edge(filter::bicubic, address::clamp_to_edge, coord::normalized);
constant sampler bicubic_zero(filter::bicubic, address::clamp_to_zero, coord::normalized);
constant sampler bicubic_white(filter::bicubic, address::clamp_to_border,
                               border_color::opaque_white, coord::normalized);
constant sampler bicubic_repeat(filter::bicubic, address::repeat, coord::normalized);
constant sampler bicubic_mirror(filter::bicubic, address::mirrored_repeat, coord::normalized);
constant sampler bicubic_pixel_edge(filter::bicubic, address::clamp_to_edge, coord::pixel);
constant sampler bicubic_pixel_zero(filter::bicubic, address::clamp_to_zero, coord::pixel);

kernel void bcu_address(texture2d<float, access::sample> t [[texture(0)]],
                        const device float4 *coord [[buffer(0)]],
                        device float4 *out [[buffer(1)]],
                        uint tid [[thread_position_in_grid]])
{
    float4 c = coord[tid];
    out[7 * tid + 0] = t.sample(bicubic_edge, c.xy);
    out[7 * tid + 1] = t.sample(bicubic_zero, c.xy);
    out[7 * tid + 2] = t.sample(bicubic_white, c.xy);
    out[7 * tid + 3] = t.sample(bicubic_repeat, c.xy);
    out[7 * tid + 4] = t.sample(bicubic_mirror, c.xy);
    out[7 * tid + 5] = t.sample(bicubic_pixel_edge, c.zw);
    out[7 * tid + 6] = t.sample(bicubic_pixel_zero, c.zw);
}
