// Owned fixture: a SIMD blur through a tile function, one tile run by a threadgroup of
// the tile's size, up to 32x32 threads. Thread (lane, simdgroup) owns the cell (lane, simdgroup),
// or (simdgroup, lane) when `transpose` is set: the simdgroup is a row (or a column) of the tile.
// It reads its texel of `source`, at `origin` plus its tile's place plus its cell, and the texel
// `fill` past it, then sums twelve taps: tap `i` is `simd_shuffle_and_fill_down` of the two by
// `i` lanes (lanes past the simdgroup's end take the next lanes' `fill` texels), weighted by
// `weights[i]`. The color is scaled by `scale` and written to the cell of an `rgba16float`
// implicit imageblock. Written for this repository; no third-party source. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct TtgBlurPixel {
    half4 color [[color(0)]];
};

struct TtgBlurParams {
    ushort2 origin;
    ushort2 fill;
    ushort transpose;
    half scale;
};

#define TTG_TAP(i) sum = fma(weights[i], simd_shuffle_and_fill_down(here, past, ushort(i)), sum)

kernel void ttg_tile_simd_blur(
    imageblock<TtgBlurPixel, imageblock_layout_implicit> block,
    texture2d<half, access::read> source [[texture(0)]],
    constant TtgBlurParams &params [[buffer(0)]],
    constant half *weights [[buffer(1)]],
    ushort2 tile [[threadgroup_position_in_grid]],
    ushort2 size [[threads_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]])
{
    ushort2 cell = params.transpose != 0 ? ushort2(simdgroup, lane) : ushort2(lane, simdgroup);
    ushort2 at = params.origin + tile * size + cell;
    half4 here = source.read(at);
    half4 past = source.read(at + params.fill);
    half4 sum = half4(0.0h);
    TTG_TAP(0); TTG_TAP(1); TTG_TAP(2); TTG_TAP(3); TTG_TAP(4); TTG_TAP(5);
    TTG_TAP(6); TTG_TAP(7); TTG_TAP(8); TTG_TAP(9); TTG_TAP(10); TTG_TAP(11);
    block.write(TtgBlurPixel{half4(sum.rgb * params.scale, sum.a)}, cell);
}
