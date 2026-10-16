// Owned fixture: programmable blending, one read-modify-write of the color
// attachment per fragment. The source color is the pixel center and two constants, blended over
// the color the fragment reads at alpha one half: `(src + dst) / 2`, one rounded sum scaled by a
// power of two, and no multiply feeds an add, so nothing can contract. A draw whose fragments
// overlap a pixel applies the blend once per fragment, each over what the one before it wrote, so
// the result pins the order Metal guarantees. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

fragment float4 exf_fetch_blend(float4 position [[position]], float4 dst [[color(0)]])
{
    float4 src = float4(position.xy, 1.0f, -0.5f);
    return (src + dst) * 0.5f;
}
