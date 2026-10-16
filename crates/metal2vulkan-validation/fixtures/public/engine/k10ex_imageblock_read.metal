// Owned fixture: a fragment that only reads its imageblock (`c`, a half4
// at byte 0, and `k`, a ushort at byte 8) and writes the color `c + k` (exact for the authored
// values). It writes no imageblock member, so it needs no fragment stores. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Cell {
    half4 c [[raster_order_group(0)]];
    ushort k [[raster_order_group(0)]];
};

fragment half4 k10ex_imageblock_read(Cell in [[imageblock_data]])
{
    return in.c + half4(half(in.k));
}
