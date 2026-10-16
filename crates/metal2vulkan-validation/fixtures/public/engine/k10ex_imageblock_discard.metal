// Owned fixture: a fragment that only writes its imageblock (`c`, a half4
// at byte 0, and `id`, a ushort at byte 8) and discards the pixels of odd `x + y`. A discarded
// fragment writes nothing, so those pixels keep the cells the pass began with; the others hold
// `c = (x, y, 0.25, -1)` and `id = 4y + x` of their pixel, and the color `(x, y, 0.5, 1)`.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Cell {
    half4 c [[raster_order_group(0)]];
    ushort id [[raster_order_group(0)]];
};

struct DiscardOut {
    half4 color [[color(0)]];
    Cell cell [[imageblock_data]];
};

fragment DiscardOut k10ex_imageblock_discard(float4 position [[position]])
{
    uint x = uint(position.x);
    uint y = uint(position.y);
    if (((x + y) & 1u) != 0u) {
        discard_fragment();
    }
    DiscardOut out;
    out.color = half4(half(x), half(y), 0.5h, 1.0h);
    out.cell.c = half4(half(x), half(y), 0.25h, -1.0h);
    out.cell.id = ushort(y * 4u + x);
    return out;
}
