// Owned fixture: partial views of a master layout whose members share
// words. `Master` holds `a` (a half at byte 0) and `b` (a half at byte 2) in word 0, `tag` (a
// uchar4) in word 1, and `n` (a ushort at byte 8) and `m` (a ushort at byte 10) in word 2. The
// fragment reads `b` and `n` through one view and writes `a`, `tag`, and `m` through another:
// `a = b + 1`, `tag = (x, y, n & 255, n >> 8)` of its pixel, `m = n + 1` (wrapping), and the color
// is `(b, n, 0, 1)`. A write of `a` or `m` that replaced its whole word would clobber `b` or `n`,
// which a later overlapping fragment of the pixel then reads. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Master {
    half a [[user(a), raster_order_group(0)]];
    half b [[user(b), raster_order_group(0)]];
    uchar4 tag [[user(tag), raster_order_group(0)]];
    ushort n [[user(n), raster_order_group(0)]];
    ushort m [[user(m), raster_order_group(0)]];
};

struct ViewIn {
    half b [[user(b)]];
    ushort n [[user(n)]];
};

struct ViewOut {
    half a [[user(a)]];
    uchar4 tag [[user(tag)]];
    ushort m [[user(m)]];
};

struct ViewsOut {
    half4 color [[color(0)]];
    ViewOut view [[imageblock_data(Master)]];
};

fragment ViewsOut k10ex_imageblock_views(ViewIn in [[imageblock_data(Master)]],
                                         float4 position [[position]])
{
    ViewsOut out;
    out.color = half4(in.b, half(in.n), 0.0h, 1.0h);
    out.view.a = in.b + 1.0h;
    out.view.tag = uchar4(uchar(position.x), uchar(position.y), uchar(in.n & 255u),
                          uchar(in.n >> 8));
    out.view.m = ushort(in.n + ushort(1));
    return out;
}
