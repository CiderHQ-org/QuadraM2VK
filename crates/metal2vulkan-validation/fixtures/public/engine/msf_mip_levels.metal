// Owned MSL for the mip-level probes: which level of a mipmapped texture a sample with an
// explicit level or a bias reads, and with what weight, through constexpr samplers that clamp to
// the edge. Each lane `i` reads one value `x[i]`, a level or a bias, chosen by the case:
//   msf_lvl2d    a `texture2d<float>`: `out[2i]` = the samples at `level(x)` through (nearest,
//                nearest mip), (nearest, linear mip), (linear, nearest mip) and (linear, linear
//                mip) at the centre, the first channel each; `out[2i + 1]` = the same through
//                (linear, linear mip) and (nearest, nearest mip) samplers clamping the level into
//                [0.3, 2.7], and through samplers minifying nearest and magnifying linearly
//                (nearest and linear mip) at texel position (5.3, 7.7) of the base level;
//   msf_bias2d   a `texture2d<float>`: `out[i]` = the four samples of `out[2i]` at `bias(x)`;
//   msf_lvl3d, msf_lvlcube, msf_lvlarray
//                a `texture3d<float>` at (0.5, 0.5, 0.5), a `texturecube<float>` along
//                (1, 0.3, 0.2), and slice 1 of a `texture2d_array<float>`: `out[i]` = the four
//                samples of `out[2i]` at `level(x)`.
// A case binds textures whose levels hold distinct values. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nn(filter::nearest, mip_filter::nearest, address::clamp_to_edge);
constant sampler nl(filter::nearest, mip_filter::linear, address::clamp_to_edge);
constant sampler ln(filter::linear, mip_filter::nearest, address::clamp_to_edge);
constant sampler ll(filter::linear, mip_filter::linear, address::clamp_to_edge);
constant sampler ll_clamped(filter::linear, mip_filter::linear, address::clamp_to_edge,
                            lod_clamp(0.3f, 2.7f));
constant sampler nn_clamped(filter::nearest, mip_filter::nearest, address::clamp_to_edge,
                            lod_clamp(0.3f, 2.7f));
constant sampler mag_n(min_filter::nearest, mag_filter::linear, mip_filter::nearest,
                       address::clamp_to_edge);
constant sampler mag_l(min_filter::nearest, mag_filter::linear, mip_filter::linear,
                       address::clamp_to_edge);

kernel void msf_lvl2d(texture2d<float> t [[texture(0)]], const device float *x [[buffer(0)]],
                      device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float l = x[i];
    float2 c = float2(0.5f);
    float2 p = float2(5.3f, 7.7f) / 16.0f;
    out[2 * i] = float4(t.sample(nn, c, level(l)).r, t.sample(nl, c, level(l)).r,
                        t.sample(ln, c, level(l)).r, t.sample(ll, c, level(l)).r);
    out[2 * i + 1] = float4(t.sample(ll_clamped, c, level(l)).r, t.sample(nn_clamped, c, level(l)).r,
                            t.sample(mag_n, p, level(l)).r, t.sample(mag_l, p, level(l)).r);
}

kernel void msf_bias2d(texture2d<float> t [[texture(0)]], const device float *x [[buffer(0)]],
                       device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float b = x[i];
    float2 c = float2(0.5f);
    out[i] = float4(t.sample(nn, c, bias(b)).r, t.sample(nl, c, bias(b)).r,
                    t.sample(ln, c, bias(b)).r, t.sample(ll, c, bias(b)).r);
}

kernel void msf_lvl3d(texture3d<float> t [[texture(0)]], const device float *x [[buffer(0)]],
                      device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float l = x[i];
    float3 c = float3(0.5f);
    out[i] = float4(t.sample(nn, c, level(l)).r, t.sample(nl, c, level(l)).r,
                    t.sample(ln, c, level(l)).r, t.sample(ll, c, level(l)).r);
}

kernel void msf_lvlcube(texturecube<float> t [[texture(0)]], const device float *x [[buffer(0)]],
                        device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float l = x[i];
    float3 d = float3(1.0f, 0.3f, 0.2f);
    out[i] = float4(t.sample(nn, d, level(l)).r, t.sample(nl, d, level(l)).r,
                    t.sample(ln, d, level(l)).r, t.sample(ll, d, level(l)).r);
}

kernel void msf_lvlarray(texture2d_array<float> t [[texture(0)]],
                         const device float *x [[buffer(0)]], device float4 *out [[buffer(1)]],
                         uint i [[thread_position_in_grid]])
{
    float l = x[i];
    float2 c = float2(0.5f);
    out[i] = float4(t.sample(nn, c, 1, level(l)).r, t.sample(nl, c, 1, level(l)).r,
                    t.sample(ln, c, 1, level(l)).r, t.sample(ll, c, 1, level(l)).r);
}
