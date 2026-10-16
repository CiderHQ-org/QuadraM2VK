// Owned fixture: samples, gathers and depth
// comparisons through the runtime sampler at [[sampler(0)]], which a case binds with
// `MirrorClampToEdge` on `s`, pixel or normalized coordinates, and nearest or linear filters.
// Each lane reads its coordinate (and a comparison's reference, `.z`) from buffer 0 and writes
// five results to buffer 1, out[5i] to out[5i + 4]: with no texel offset, then with the offsets
// -3, 2, -7 and 5 on `s`. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void rrp_mc_sample(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                          const device float4 *coord [[buffer(0)]],
                          device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float2 p = coord[i].xy;
    out[5 * i + 0] = t.sample(s, p);
    out[5 * i + 1] = t.sample(s, p, int2(-3, 0));
    out[5 * i + 2] = t.sample(s, p, int2(2, 0));
    out[5 * i + 3] = t.sample(s, p, int2(-7, 0));
    out[5 * i + 4] = t.sample(s, p, int2(5, 0));
}

kernel void rrp_mc_gather(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                          const device float4 *coord [[buffer(0)]],
                          device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float2 p = coord[i].xy;
    out[5 * i + 0] = t.gather(s, p);
    out[5 * i + 1] = t.gather(s, p, int2(-3, 0));
    out[5 * i + 2] = t.gather(s, p, int2(2, 0));
    out[5 * i + 3] = t.gather(s, p, int2(-7, 0));
    out[5 * i + 4] = t.gather(s, p, int2(5, 0));
}

kernel void rrp_mc_compare(depth2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                           const device float4 *coord [[buffer(0)]],
                           device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float2 p = coord[i].xy;
    float r = coord[i].z;
    out[5 * i + 0] = float4(t.sample_compare(s, p, r), 0.0, 0.0, 0.0);
    out[5 * i + 1] = float4(t.sample_compare(s, p, r, int2(-3, 0)), 0.0, 0.0, 0.0);
    out[5 * i + 2] = float4(t.sample_compare(s, p, r, int2(2, 0)), 0.0, 0.0, 0.0);
    out[5 * i + 3] = float4(t.sample_compare(s, p, r, int2(-7, 0)), 0.0, 0.0, 0.0);
    out[5 * i + 4] = float4(t.sample_compare(s, p, r, int2(5, 0)), 0.0, 0.0, 0.0);
}
