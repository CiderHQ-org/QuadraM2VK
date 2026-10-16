// Owned fixture: three color attachments of three numeric classes (float,
// unsigned integer, half) read and written by one fragment. Each output mixes values the fragment
// read from the others: `color` averages itself with `tint` (one rounded sum scaled by a power of
// two; no multiply feeds an add, so nothing can contract), `tag` adds the bits of `color.x` to
// five times itself (wrapping), and `tint` rotates its lanes and doubles them (exact). Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Three {
    float4 color [[color(0)]];
    uint tag [[color(1)]];
    half4 tint [[color(2)]];
};

fragment Three exf_fetch_three(float4 color [[color(0)]],
                               uint tag [[color(1)]],
                               half4 tint [[color(2)]])
{
    Three out;
    out.color = (color + float4(tint)) * 0.5f;
    out.tag = tag * 5u + as_type<uint>(color.x);
    out.tint = tint.yzwx * 2.0h;
    return out;
}
