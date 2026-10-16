// Owned MSL: the two sampling strategies no case ran, over
// a mipmapped texture whose levels differ. Written from scratch for this repository; loop-free.
//
// mip_bias_kernel: a kernel's sample with no level or gradient. Metal reads its default level 0
// shifted by the bias (`sampler.sample_base_level`); each thread takes its
// coordinate and bias from the input buffer.
//
// mip_implicit_fragment: a fragment function's implicit-level sample (`sampler.sample_implicit`).
// Each 2x2 quad of the target scales its pixel position by its own power of two, 2^q / 16 for
// the q-th quad along x, so the coordinate's derivatives are exactly 2^q / 16 per pixel and, on a
// 16-texel-wide texture, the level of detail is exactly q.
#include <metal_stdlib>
using namespace metal;

kernel void mip_bias_kernel(texture2d<float> t [[texture(0)]],
                            sampler s [[sampler(0)]],
                            const device float4 *params [[buffer(0)]],
                            device float4 *out [[buffer(1)]],
                            uint gid [[thread_position_in_grid]])
{
    float4 q = params[gid];
    out[gid] = t.sample(s, q.xy, bias(q.z));
}

fragment float4 mip_implicit_fragment(float4 pos [[position]],
                                      texture2d<float> t [[texture(0)]],
                                      sampler s [[sampler(0)]])
{
    float scale = float(1u << (uint(pos.x) >> 1)) / 16.0f;
    return t.sample(s, pos.xy * scale);
}
