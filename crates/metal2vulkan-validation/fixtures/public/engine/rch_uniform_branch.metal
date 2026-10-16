// Owned fixture: a fragment function that samples a texture through a
// runtime sampler at a coordinate it divides by the texture's size where a constant block's flag
// says, a branch every invocation of a draw takes alike, and writes a buffer. Under the sampler's
// YCbCr variant the emulated linear filter branches on the sampler's specialization
// constants, and its device-sample arm converts a `half` texel through a rounded add.
#include <metal_stdlib>
using namespace metal;

// Where `rch_uniform_branch` samples: `origin + g * step` at pixel `g`, divided by the texture's
// size where `normalized` is not 0, written at `g.y * columns + g.x`.
struct UniformProbe {
    float2 origin;
    float2 step;
    uint columns;
    uint normalized;
};

fragment float4 rch_uniform_branch(float4 position [[position]],
                                   texture2d<half> t [[texture(0)]],
                                   sampler s [[sampler(0)]],
                                   device half4 *out [[buffer(0)]],
                                   constant UniformProbe &p [[buffer(1)]])
{
    uint2 g = uint2(position.xy);
    float2 at = p.origin + float2(g) * p.step;
    if (p.normalized != 0) {
        at /= float2(t.get_width(), t.get_height());
    }
    out[g.y * p.columns + g.x] = t.sample(s, at);
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}
