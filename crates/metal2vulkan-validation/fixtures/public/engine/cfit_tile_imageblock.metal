// Owned fixture: a render pass's imageblock shared by its fragment
// functions' `[[imageblock_data]]` and its tile functions' explicit imageblock, in the splat
// rasterizer's shape: a tile function clears each pixel's cell, each draw's fragments set their
// splat's bit in their pixel's cell under a raster order group, and a tile function resolves the
// cells into a buffer. Metal keeps the imageblock in tile memory, each pixel's same bytes for
// every layout of one size, for the whole pass: a dispatch reads what the draws and dispatches
// before it wrote, and the cells read zero as a pass begins (measured on an M4). The fragment
// functions write one `r32uint` color attachment, which the tile pipelines name; a tile function
// runs one thread per pixel of its tile, its cell `thread_position_in_threadgroup`. A layout
// aliased onto the attachment, the tile's implicit imageblock, is a refusal's subject. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Corner {
    float4 position [[position]];
};

// One corner of the rectangle of pixels `rect.xy` to `rect.zw` (excluded) in a `size` render
// area, drawn as a strip of four vertices.
vertex Corner cfit_ib_rect(uint vid [[vertex_id]],
                           constant float4 &rect [[buffer(0)]],
                           constant float2 &size [[buffer(1)]])
{
    float2 pixel = float2((vid & 1u) != 0u ? rect.z : rect.x, (vid & 2u) != 0u ? rect.w : rect.y);
    Corner out;
    out.position = float4(pixel.x / size.x * 2.0f - 1.0f, 1.0f - pixel.y / size.y * 2.0f, 0.0f, 1.0f);
    return out;
}

// The fragments' cell: 64 bits of splats, bit `s` the splat `s`.
struct Bits {
    uint lo [[raster_order_group(0)]];
    uint hi [[raster_order_group(0)]];
};

struct BitsOut {
    uint color [[color(0)]];
    Bits cell [[imageblock_data]];
};

// The cell gains bit `splat` (below 64); the pixel takes the cell's low word.
fragment BitsOut cfit_ib_splat(Corner in [[stage_in]],
                               Bits cell [[imageblock_data]],
                               constant uint &splat [[buffer(0)]])
{
    BitsOut out;
    out.cell = cell;
    if (splat < 32u) {
        out.cell.lo = cell.lo | (1u << splat);
    } else {
        out.cell.hi = cell.hi | (1u << (splat - 32u));
    }
    out.color = out.cell.lo;
    return out;
}

// The tile functions' cell: the same eight bytes.
struct TileBits {
    uint lo;
    uint hi;
};

// The same bytes through another layout of the same size.
struct TileHalves {
    ushort a;
    ushort b;
    uint z;
};

// Every cell takes `seed`.
kernel void cfit_ib_tile_clear(imageblock<TileBits, imageblock_layout_explicit> block,
                               constant uint2 &seed [[buffer(1)]],
                               ushort2 t [[thread_position_in_threadgroup]])
{
    threadgroup_imageblock TileBits *cell = block.data(t);
    cell->lo = seed.x;
    cell->hi = seed.y;
}

// Every cell's low word gains `add`.
kernel void cfit_ib_tile_add(imageblock<TileBits, imageblock_layout_explicit> block,
                             constant uint &add [[buffer(1)]],
                             ushort2 t [[thread_position_in_threadgroup]])
{
    threadgroup_imageblock TileBits *cell = block.data(t);
    cell->lo = cell->lo + add;
}

// out[2 * i] and out[2 * i + 1] take the low and high words of pixel `i = y * size.x + x`'s cell,
// for the pixels inside the `size` render area.
kernel void cfit_ib_tile_resolve(imageblock<TileBits, imageblock_layout_explicit> block,
                                 device uint *out [[buffer(0)]],
                                 constant uint2 &size [[buffer(1)]],
                                 ushort2 t [[thread_position_in_threadgroup]],
                                 ushort2 g [[threadgroup_position_in_grid]],
                                 ushort2 n [[threads_per_threadgroup]])
{
    TileBits cell = *block.data(t);
    uint2 p = uint2(g) * uint2(n) + uint2(t);
    if (p.x < size.x && p.y < size.y) {
        uint i = p.y * size.x + p.x;
        out[2u * i] = cell.lo;
        out[2u * i + 1u] = cell.hi;
    }
}

// As `cfit_ib_tile_resolve`, through the layout of two halves and a word: out[2 * i] takes
// `b << 16 | a`, out[2 * i + 1] takes `z`.
kernel void cfit_ib_tile_resolve_halves(imageblock<TileHalves, imageblock_layout_explicit> block,
                                        device uint *out [[buffer(0)]],
                                        constant uint2 &size [[buffer(1)]],
                                        ushort2 t [[thread_position_in_threadgroup]],
                                        ushort2 g [[threadgroup_position_in_grid]],
                                        ushort2 n [[threads_per_threadgroup]])
{
    TileHalves cell = *block.data(t);
    uint2 p = uint2(g) * uint2(n) + uint2(t);
    if (p.x < size.x && p.y < size.y) {
        uint i = p.y * size.x + p.x;
        out[2u * i] = (uint(cell.b) << 16) | uint(cell.a);
        out[2u * i + 1u] = cell.z;
    }
}

// The attachment's texel, through a layout aliased onto the tile's implicit imageblock.
struct Texel {
    uint color;
};

// Every pixel's texel takes `v`.
kernel void cfit_ib_tile_aliased(imageblock<Texel, imageblock_layout_explicit> block [[alias_implicit_imageblock]],
                                 constant uint &v [[buffer(1)]],
                                 ushort2 t [[thread_position_in_threadgroup]])
{
    threadgroup_imageblock Texel *cell = block.data(t);
    cell->color = v;
}
