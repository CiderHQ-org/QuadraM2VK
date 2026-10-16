// Owned fixture: Metal's comparison filtering of depth textures, so that a translation
// can emulate it in the shader. Every lane `i` reads a job `job[i]`: a normalized coordinate (or a
// cube direction) and a reference value, chosen by the case, and compares through constexpr
// samplers that clamp to the edge, with normalized coordinates and no mip filter:
//   msf_cmp_sample2d  a `depth2d<float>`: `out[i]` = the `sample_compare` of a linear and of a
//                     nearest sampler, each under `less_equal` and `equal`, in the order
//                     (linear less_equal, linear equal, nearest less_equal, nearest equal);
//   msf_cmp_gather2d  a `depth2d<float>`: `out[3i]` = the plain `gather` of the four texels,
//                     `out[3i + 1]` and `out[3i + 2]` = `gather_compare` under `less_equal` and
//                     under `equal`;
//   msf_cmp_cube      a `depthcube<float>`: `out[i]` = the `sample_compare` of the linear sampler
//                     under `less_equal` and `equal`, of the nearest one under `equal`, and the
//                     plain linear `sample` of the depth.
// The cases bind textures whose texels hold distinct values, so a comparison under `equal` against
// one texel's value isolates that texel's weight. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler lin_le(filter::linear, address::clamp_to_edge, compare_func::less_equal);
constant sampler lin_eq(filter::linear, address::clamp_to_edge, compare_func::equal);
constant sampler near_le(filter::nearest, address::clamp_to_edge, compare_func::less_equal);
constant sampler near_eq(filter::nearest, address::clamp_to_edge, compare_func::equal);
constant sampler plain_near(filter::nearest, address::clamp_to_edge);
constant sampler plain_lin(filter::linear, address::clamp_to_edge);

kernel void msf_cmp_sample2d(depth2d<float, access::sample> t [[texture(0)]],
                             const device float4 *job [[buffer(0)]],
                             device float4 *out [[buffer(1)]],
                             uint i [[thread_position_in_grid]])
{
    float4 j = job[i];
    float2 u = j.xy;
    float r = j.z;
    out[i] = float4(t.sample_compare(lin_le, u, r), t.sample_compare(lin_eq, u, r),
                    t.sample_compare(near_le, u, r), t.sample_compare(near_eq, u, r));
}

kernel void msf_cmp_gather2d(depth2d<float, access::sample> t [[texture(0)]],
                             const device float4 *job [[buffer(0)]],
                             device float4 *out [[buffer(1)]],
                             uint i [[thread_position_in_grid]])
{
    float4 j = job[i];
    float2 u = j.xy;
    float r = j.z;
    device float4 *o = out + 3 * i;
    o[0] = t.gather(plain_near, u);
    o[1] = t.gather_compare(near_le, u, r);
    o[2] = t.gather_compare(near_eq, u, r);
}

kernel void msf_cmp_cube(depthcube<float, access::sample> c [[texture(0)]],
                         const device float4 *job [[buffer(0)]],
                         device float4 *out [[buffer(1)]],
                         uint i [[thread_position_in_grid]])
{
    float4 j = job[i];
    float3 d = j.xyz;
    float r = j.w;
    out[i] = float4(c.sample_compare(lin_le, d, r), c.sample_compare(lin_eq, d, r),
                    c.sample_compare(near_eq, d, r), c.sample(plain_lin, d));
}
