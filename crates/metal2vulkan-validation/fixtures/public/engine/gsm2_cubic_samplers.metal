// Owned fixture (written for this repository): a sample through the runtime
// sampler at slot 0 beside the same sample through a constexpr bicubic sampler of each
// address mode, border color, and coordinate kind. Metal samples through a runtime sampler whose
// min or mag filter is `MTLSamplerMinMagFilterCubic` (raw value 2) as through a constexpr bicubic
// sampler of the same state, so the runtime lane binds the slot to cubic
// samplers of each state and requires every lane of the two samples to be the same bits.
#include <metal_stdlib>
using namespace metal;

constexpr sampler bc_edge(filter::bicubic, address::clamp_to_edge);
constexpr sampler bc_repeat(filter::bicubic, address::repeat);
constexpr sampler bc_mirror(filter::bicubic, address::mirrored_repeat);
constexpr sampler bc_zero(filter::bicubic, address::clamp_to_zero);
constexpr sampler bc_white(filter::bicubic, address::clamp_to_border, border_color::opaque_white);
constexpr sampler bc_black(filter::bicubic, address::clamp_to_border, border_color::opaque_black);
constexpr sampler bc_pixel(filter::bicubic, address::clamp_to_edge, coord::pixel);

// `which` names the constexpr sampler: 0 edge, 1 repeat, 2 mirrored repeat, 3 zero, 4 an
// opaque-white border, 5 an opaque-black border, 6 pixel coordinates clamped to the edge. Lane
// `i` writes the runtime sampler's sample, then the constexpr one's.
kernel void gsm2_cubic_samples(texture2d<float> t [[texture(0)]],
                               sampler s [[sampler(0)]],
                               device const float2 *uv [[buffer(0)]],
                               device float4 *out [[buffer(1)]],
                               constant uint &which [[buffer(2)]],
                               uint i [[thread_position_in_grid]])
{
    float2 p = uv[i];
    float4 c;
    switch (which) {
    case 0: c = t.sample(bc_edge, p); break;
    case 1: c = t.sample(bc_repeat, p); break;
    case 2: c = t.sample(bc_mirror, p); break;
    case 3: c = t.sample(bc_zero, p); break;
    case 4: c = t.sample(bc_white, p); break;
    case 5: c = t.sample(bc_black, p); break;
    default: c = t.sample(bc_pixel, p); break;
    }
    out[2 * i] = t.sample(s, p);
    out[2 * i + 1] = c;
}

// The same with `half` results at an explicit level, which a bicubic sample does not read (its
// filter samples the base level): lane `i` writes the runtime sampler's sample at level
// 2.5, then the constexpr edge sampler's implicit one.
kernel void gsm2_cubic_half_samples(texture2d<half> t [[texture(0)]],
                                    sampler s [[sampler(0)]],
                                    device const float2 *uv [[buffer(0)]],
                                    device half4 *out [[buffer(1)]],
                                    uint i [[thread_position_in_grid]])
{
    float2 p = uv[i];
    out[2 * i] = t.sample(s, p, level(2.5f));
    out[2 * i + 1] = t.sample(bc_edge, p);
}
