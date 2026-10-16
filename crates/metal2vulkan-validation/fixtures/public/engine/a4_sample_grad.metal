// Owned fixture: a kernel sample with explicit gradients
// (`sampler.sample_grad`). Thread g samples an 8 x 1 texture at the centre of texel g; through a
// nearest sampler of a texture of one level the gradients choose no other texel:
// out[g] = t(g, 0). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_sample_grad(texture2d<float> t [[texture(0)]],
                           sampler s [[sampler(0)]],
                           device float4 *out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    float2 uv = (float2(gid, 0) + 0.5) * float2(0.125, 1.0);
    out[gid] = t.sample(s, uv, gradient2d(float2(0.25, 0.0), float2(0.0, 0.25)));
}
