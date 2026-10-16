// Owned MSL for the runtime-sampler level probes: the level, and the texel at that level, a sample
// at an explicit level or a kernel's bias reads through a sampler the function learns only at run
// time.
//   unv_rlvl2d  a `texture2d<float>` sampled through the runtime sampler at slot 0, and the
//               argument buffer's texture through the argument buffer's sampler, at the centre:
//               `out[i]` = the slot's samples at `level(x[i])` and `bias(x[i])`, then the argument
//               buffer's at `level(x[i])` and `bias(x[i])`, the first channel each;
//   unv_rpos2d  a `texture2d<float>` sampled through the runtime sampler at slot 0: `out[i]` = the
//               sample at `(p[i].x, p[i].y)` at `level(p[i].z)`, the first channel.
// A case binds textures whose levels hold distinct values. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Material {
    texture2d<float> t;
    sampler s;
};

kernel void unv_rlvl2d(texture2d<float> t [[texture(0)]], sampler r [[sampler(0)]],
                       constant Material &m [[buffer(0)]], const device float *x [[buffer(1)]],
                       device float4 *out [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    float l = x[i];
    float2 c = float2(0.5f);
    out[i] = float4(t.sample(r, c, level(l)).r, t.sample(r, c, bias(l)).r,
                    m.t.sample(m.s, c, level(l)).r, m.t.sample(m.s, c, bias(l)).r);
}

kernel void unv_rpos2d(texture2d<float> t [[texture(0)]], sampler r [[sampler(0)]],
                       const device float4 *p [[buffer(0)]], device float *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    float4 v = p[i];
    out[i] = t.sample(r, v.xy, level(v.z)).r;
}
