// Owned fixture: a fragment function that reads no attachment and writes
// color attachments 0 and 2 but not color attachment 1, between them: the pixel center and two
// constants to attachment 0, the pixel's integer coordinates and two constants to attachment 2.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct SkipOne {
    float4 color [[color(0)]];
    uint4 mark [[color(2)]];
};

fragment SkipOne exf2_skip_one(float4 position [[position]])
{
    SkipOne out;
    out.color = float4(position.xy, 1.0f, -0.5f);
    out.mark = uint4(uint(position.x), uint(position.y), 7u, 0xFFFFFFFFu);
    return out;
}
