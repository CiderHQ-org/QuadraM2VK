// Owned fixture: Metal's bicubic filter in a fragment function, the corpus's
// other form: an implicit sample whose level of detail comes from the coordinate's derivatives.
// The sampler has no mip filter and filters bicubic both ways, so the level should only choose
// between two identical filters; the cases magnify (a quarter texel per pixel) and minify (two
// texels per pixel). The coordinate is `pos.xy * p.xy + p.zw` with dyadic `p`, so it is exact
// whether or not the compiler fuses the multiply-add. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler bicubic_edge(filter::bicubic, address::clamp_to_edge, coord::normalized);

fragment float4 bcu_fragment(float4 pos [[position]],
                             texture2d<float, access::sample> t [[texture(0)]],
                             constant float4 &p [[buffer(0)]])
{
    return t.sample(bicubic_edge, pos.xy * p.xy + p.zw);
}
