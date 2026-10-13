// Owned fixture: 2D bicubic samples of an extended-range texture (`BGR10_XR`, `BGRA10_XR`,
// `RGB10A8_2P_XR10` or an `_sRGB` twin), one thread per coordinate: through the runtime sampler at
// `[[sampler(0)]]`, and through constexpr bicubic samplers under each address mode and border
// color, with pixel coordinates, and with a constant texel offset, with `float` results and,
// through the runtime sampler and clamp to edge, `half` ones. Loop-free.
#include <metal_stdlib>
using namespace metal;

constexpr sampler xrb_edge_sampler(filter::bicubic, address::clamp_to_edge);
constexpr sampler xrb_repeat_sampler(filter::bicubic, address::repeat);
constexpr sampler xrb_mirror_sampler(filter::bicubic, address::mirrored_repeat);
constexpr sampler xrb_zero_sampler(filter::bicubic, address::clamp_to_zero);
constexpr sampler xrb_white_sampler(filter::bicubic, address::clamp_to_border,
                                    border_color::opaque_white);
constexpr sampler xrb_black_sampler(filter::bicubic, address::clamp_to_border,
                                    border_color::opaque_black);
constexpr sampler xrb_pixel_sampler(filter::bicubic, address::clamp_to_edge, coord::pixel);

kernel void xrb_runtime(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                        device const float2 *uv [[buffer(0)]], device float4 *out [[buffer(1)]],
                        uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(s, uv[g]);
}

kernel void xrb_runtime_half(texture2d<half> t [[texture(0)]], sampler s [[sampler(0)]],
                             device const float2 *uv [[buffer(0)]],
                             device half4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(s, uv[g]);
}

kernel void xrb_edge(texture2d<float> t [[texture(0)]], device const float2 *uv [[buffer(0)]],
                     device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_edge_sampler, uv[g]);
}

kernel void xrb_edge_half(texture2d<half> t [[texture(0)]], device const float2 *uv [[buffer(0)]],
                          device half4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_edge_sampler, uv[g]);
}

kernel void xrb_repeat(texture2d<float> t [[texture(0)]], device const float2 *uv [[buffer(0)]],
                       device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_repeat_sampler, uv[g]);
}

kernel void xrb_mirror(texture2d<float> t [[texture(0)]], device const float2 *uv [[buffer(0)]],
                       device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_mirror_sampler, uv[g]);
}

kernel void xrb_zero(texture2d<float> t [[texture(0)]], device const float2 *uv [[buffer(0)]],
                     device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_zero_sampler, uv[g]);
}

kernel void xrb_white(texture2d<float> t [[texture(0)]], device const float2 *uv [[buffer(0)]],
                      device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_white_sampler, uv[g]);
}

kernel void xrb_black(texture2d<float> t [[texture(0)]], device const float2 *uv [[buffer(0)]],
                      device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_black_sampler, uv[g]);
}

kernel void xrb_pixel(texture2d<float> t [[texture(0)]], device const float2 *uv [[buffer(0)]],
                      device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_pixel_sampler, uv[g] * float2(t.get_width(), t.get_height()));
}

kernel void xrb_edge_offset(texture2d<float> t [[texture(0)]],
                            device const float2 *uv [[buffer(0)]], device float4 *out [[buffer(1)]],
                            uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_edge_sampler, uv[g], int2(1, -2));
}

kernel void xrb_repeat_offset(texture2d<float> t [[texture(0)]],
                              device const float2 *uv [[buffer(0)]],
                              device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(xrb_repeat_sampler, uv[g], int2(1, -2));
}
