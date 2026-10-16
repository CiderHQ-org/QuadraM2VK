// Owned fixture: `clamp_to_border` with each Metal border colour per bound
// pixel format (from earlier border and alpha observations). Vulkan's
// border colours are RGBA constants, so what a format without alpha, or with fewer channels,
// returns at the border is a reading. One float, one uint, and one int texture are sampled at
// coordinates inside, outside, and on each edge through constexpr nearest samplers with the
// transparent-black, opaque-black, and opaque-white borders, plus a linear opaque-white one for
// the float texture. The cases bind different formats to the same three slots. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler border_transparent(filter::nearest, address::clamp_to_border,
                                    border_color::transparent_black, coord::normalized);
constant sampler border_black(filter::nearest, address::clamp_to_border,
                              border_color::opaque_black, coord::normalized);
constant sampler border_white(filter::nearest, address::clamp_to_border,
                              border_color::opaque_white, coord::normalized);
constant sampler border_white_linear(filter::linear, address::clamp_to_border,
                                     border_color::opaque_white, coord::normalized);

kernel void a2_border_color(texture2d<float, access::sample> f [[texture(0)]],
                            texture2d<uint, access::sample> u [[texture(1)]],
                            texture2d<int, access::sample> i [[texture(2)]],
                            const device float2 *coord [[buffer(0)]],
                            device uint4 *out [[buffer(1)]],
                            uint tid [[thread_position_in_grid]])
{
    float2 uv = coord[tid];
    out[10 * tid + 0] = as_type<uint4>(f.sample(border_transparent, uv, level(0)));
    out[10 * tid + 1] = as_type<uint4>(f.sample(border_black, uv, level(0)));
    out[10 * tid + 2] = as_type<uint4>(f.sample(border_white, uv, level(0)));
    out[10 * tid + 3] = as_type<uint4>(f.sample(border_white_linear, uv, level(0)));
    out[10 * tid + 4] = u.sample(border_transparent, uv, level(0));
    out[10 * tid + 5] = u.sample(border_black, uv, level(0));
    out[10 * tid + 6] = u.sample(border_white, uv, level(0));
    out[10 * tid + 7] = as_type<uint4>(i.sample(border_transparent, uv, level(0)));
    out[10 * tid + 8] = as_type<uint4>(i.sample(border_black, uv, level(0)));
    out[10 * tid + 9] = as_type<uint4>(i.sample(border_white, uv, level(0)));
}
