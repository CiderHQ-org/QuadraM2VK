// Owned fixture: `clamp_to_zero` addressing per bound pixel format.
// Metal documents the out-of-range texel as (0, 0, 0, 0) for a format with alpha
// and (0, 0, 0, 1) without; which formats count as having alpha, and what a linear blend with that
// zero gives, are readings. One float, one uint, and one int texture are sampled at coordinates
// inside, outside, and on each edge: through constexpr nearest and linear clamp-to-zero samplers
// and through the runtime sampler at [[sampler(0)]] (which the case sets to nearest clamp-to-zero).
// The cases bind different formats to the same three slots. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler zero_nearest(filter::nearest, address::clamp_to_zero, coord::normalized);
constant sampler zero_linear(filter::linear, address::clamp_to_zero, coord::normalized);

kernel void a2_clamp_to_zero(texture2d<float, access::sample> f [[texture(0)]],
                             texture2d<uint, access::sample> u [[texture(1)]],
                             texture2d<int, access::sample> i [[texture(2)]],
                             sampler runtime_sampler [[sampler(0)]],
                             const device float2 *coord [[buffer(0)]],
                             device uint4 *out [[buffer(1)]],
                             uint tid [[thread_position_in_grid]])
{
    float2 uv = coord[tid];
    out[5 * tid + 0] = as_type<uint4>(f.sample(zero_nearest, uv, level(0)));
    out[5 * tid + 1] = as_type<uint4>(f.sample(zero_linear, uv, level(0)));
    out[5 * tid + 2] = as_type<uint4>(f.sample(runtime_sampler, uv, level(0)));
    out[5 * tid + 3] = u.sample(zero_nearest, uv, level(0));
    out[5 * tid + 4] = as_type<uint4>(i.sample(zero_nearest, uv, level(0)));
}
