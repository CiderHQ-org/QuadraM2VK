// Owned fixture: a kernel that samples one element of a 32-texture array,
// picked per thread, through one runtime sampler. Where 32 combined image samplers exceed the
// stage's samplers the stage samples 32 separate textures through one separate sampler, which
// the binder writes with the sampler bound at slot 0 made for the use; elsewhere the 32 combined
// bindings each take that sampler. Thread i samples texture i & 31, of one texel, at its centre
// at level 0: out[i] = that texel. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void smp_runtime_array(array<texture2d<float>, 32> t [[texture(0)]],
                              sampler s [[sampler(0)]],
                              device float4 *out [[buffer(0)]],
                              uint gid [[thread_position_in_grid]]) {
    out[gid] = t[gid & 31].sample(s, float2(0.5), level(0));
}
