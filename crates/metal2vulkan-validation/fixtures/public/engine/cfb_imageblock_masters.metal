// Owned fixture: a fragment function whose explicit imageblock is another master of the eight-byte
// splat cell of `cfit_tile_imageblock` (two `uint`s, aligned to 4), its words swapped, so a lane
// draws both masters in one pass as a Metal probe did: Metal's views of masters of one size and
// alignment reach the same bytes, each at its own offsets. The vertex function is `cfit_ib_rect`'s
// rectangle. Written for this repository; no third-party source. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct CfbIbCorner {
    float4 position [[position]];
};

// One corner of the rectangle of pixels `rect.xy` to `rect.zw` (excluded) in a `size` render
// area, drawn as a strip of four vertices.
vertex CfbIbCorner cfb_ib_rect(uint vid [[vertex_id]],
                               constant float4 &rect [[buffer(0)]],
                               constant float2 &size [[buffer(1)]])
{
    float2 pixel = float2((vid & 1u) != 0u ? rect.z : rect.x, (vid & 2u) != 0u ? rect.w : rect.y);
    CfbIbCorner out;
    out.position = float4(pixel.x / size.x * 2.0f - 1.0f, 1.0f - pixel.y / size.y * 2.0f, 0.0f, 1.0f);
    return out;
}

// The splat cell's eight bytes through another master: `hi` first, `lo` second.
struct CfbIbSwapped {
    uint hi [[raster_order_group(0)]];
    uint lo [[raster_order_group(0)]];
};

struct CfbIbSwappedOut {
    uint color [[color(0)]];
    CfbIbSwapped cell [[imageblock_data]];
};

// The cell's `lo`, its second word, gains bit `splat` (below 32); the pixel takes it.
fragment CfbIbSwappedOut cfb_ib_splat_swapped(CfbIbCorner in [[stage_in]],
                                              CfbIbSwapped cell [[imageblock_data]],
                                              constant uint &splat [[buffer(0)]])
{
    CfbIbSwappedOut out;
    out.cell = cell;
    out.cell.lo = cell.lo | (1u << splat);
    out.color = out.cell.lo;
    return out;
}
