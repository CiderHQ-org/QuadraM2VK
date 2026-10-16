// Owned fixture (written for this repository): 2D linear
// samples of an extended-range texture (`BGRA10_XR`, `BGR10_XR` or an `_sRGB` twin), through the
// runtime sampler at `[[sampler(0)]]` and through a constexpr linear clamp-to-edge sampler, with
// `float` and `half` results. One thread per coordinate. Loop-free.
#include <metal_stdlib>
using namespace metal;

constexpr sampler edge(filter::linear, address::clamp_to_edge, coord::normalized);

// The sample of `t` through the runtime sampler `s` at `uv[g]`.
kernel void xr_sample_runtime(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                              device const float2 *uv [[buffer(0)]],
                              device float4 *out [[buffer(1)]],
                              uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(s, uv[g]);
}

kernel void xr_sample_runtime_half(texture2d<half> t [[texture(0)]], sampler s [[sampler(0)]],
                                   device const float2 *uv [[buffer(0)]],
                                   device half4 *out [[buffer(1)]],
                                   uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(s, uv[g]);
}

// The sample of `t` through the constexpr sampler `edge` at `uv[g]`.
kernel void xr_sample_constexpr(texture2d<float> t [[texture(0)]],
                                device const float2 *uv [[buffer(0)]],
                                device float4 *out [[buffer(1)]],
                                uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(edge, uv[g]);
}

kernel void xr_sample_constexpr_half(texture2d<half> t [[texture(0)]],
                                     device const float2 *uv [[buffer(0)]],
                                     device half4 *out [[buffer(1)]],
                                     uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(edge, uv[g]);
}
