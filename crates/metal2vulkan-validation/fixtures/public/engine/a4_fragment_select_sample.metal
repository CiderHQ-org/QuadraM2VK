// Owned fixture: an implicit-LOD sample through a texture selected per
// fragment (`sampler.sample_hoisted_grad.switch`: the gradients are hoisted before the switch).
// Pixel (x, y) of a 16 x 16 target samples a when its centre's x exceeds 8, else b,
// at the centre of texel (x, y); through a nearest sampler of 16 x 16 textures of one level that
// is texel (x, y) of the chosen texture. Loop-free.
#include <metal_stdlib>
using namespace metal;

fragment float4 a4_fragment_select_sample(float4 pos [[position]],
                                          texture2d<float> a [[texture(0)]],
                                          texture2d<float> b [[texture(1)]],
                                          sampler s [[sampler(0)]]) {
    texture2d<float> t = pos.x > 8.0 ? a : b;
    return t.sample(s, pos.xy * 0.0625);
}
