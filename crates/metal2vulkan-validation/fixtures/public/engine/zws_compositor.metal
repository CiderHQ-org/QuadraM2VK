// Owned fixture: a compositor's rounded, half-transparent boxes drawn over what a
// surface already holds, as a window server draws alert windows over a wallpaper. `zws_box`
// places two triangles over a box given in pixels of a target of the given size. Both fragment
// functions keep the fragments inside the box's rounded corners (a circle of the radius at each
// corner, tested at the pixel center: half-integers squared, exact, never on the circle).
// `zws_over` blends by programmable blending: the premultiplied color plus the color it reads of
// the attachment times `keep`, a power of two in every box the lanes draw, so each product is
// exact and only the sum rounds (nothing a contraction could change). `zws_plain` returns the
// color, for the fixed-function blend. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct ZwsBox {
    float4 rect;  // x0, y0, x1, y1 in pixels
    float4 color; // premultiplied
    float4 keep;  // what of the attachment's color stays, each channel
    float4 shape; // corner radius, target width, target height, unused
};

struct ZwsOut {
    float4 position [[position]];
};

vertex ZwsOut zws_box(uint vid [[vertex_id]], constant ZwsBox &box [[buffer(0)]])
{
    const float2 corners[6] = {
        float2(0.0f, 0.0f), float2(1.0f, 0.0f), float2(0.0f, 1.0f),
        float2(1.0f, 0.0f), float2(1.0f, 1.0f), float2(0.0f, 1.0f),
    };
    float2 t = corners[vid];
    float2 p = select(box.rect.xy, box.rect.zw, t > 0.5f);
    float2 size = box.shape.yz;
    ZwsOut out;
    out.position = float4(p.x / size.x * 2.0f - 1.0f, 1.0f - p.y / size.y * 2.0f, 0.0f, 1.0f);
    return out;
}

// Whether the pixel center `p` lies outside `box`'s rounded corners.
static bool zws_outside(float2 p, constant ZwsBox &box)
{
    float r = box.shape.x;
    float2 q = p - clamp(p, box.rect.xy + r, box.rect.zw - r);
    return q.x * q.x + q.y * q.y > r * r;
}

fragment float4 zws_over(ZwsOut in [[stage_in]], float4 dst [[color(0)]],
                         constant ZwsBox &box [[buffer(0)]])
{
    if (zws_outside(in.position.xy, box)) {
        discard_fragment();
    }
    return box.color + dst * box.keep;
}

fragment float4 zws_plain(ZwsOut in [[stage_in]], constant ZwsBox &box [[buffer(0)]])
{
    if (zws_outside(in.position.xy, box)) {
        discard_fragment();
    }
    return box.color;
}
