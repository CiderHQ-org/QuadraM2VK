// Owned fixture: a fragment imageblock write beside a returned
// sample mask, and views of two masters in one function, as Metal answers them (measured on the
// M3 Max and an M4).
//
// `cover_masked` writes the whole `Cell` of each pixel and returns a sample mask by pixel class
// (x + 2y) % 4: 0, 1, 0xfffffffe, 0xffffffff. Metal drops the fragment, and its imageblock write,
// where bit 0 is clear (classes 0 and 2), whatever the other bits.
//
// The two-master functions view `Wide` (8 bytes: `a` at byte 0, `b` at byte 2, `w` at byte 4) and
// a smaller master whose members of the same names lie at other bytes. Each view reaches its own
// master's bytes of the one cell, which spans `Wide`: `cover_two_down` reads `a` through `Wide`
// (byte 0) and writes `a` through `Swapped` (byte 2, `Wide`'s `b`); `cover_two_up` reads `a`
// through `Swapped` (byte 2) and writes `a` through `Wide` (byte 0); `cover_two_word` reads the
// first word through `Word` (`w` at byte 0, `Wide`'s `a` and `b`) and writes its high half as
// `Wide`'s `a` and its low half as `Wide`'s `w`.
//
// Each function also writes the color (x, y, 0.5, 1) of its pixel to attachment 0, which a case's
// pass needs for its extent. `cover_vertex` draws a full-viewport triangle for render pipelines of
// these functions.
#include <metal_stdlib>
using namespace metal;

struct Corner {
    float4 position [[position]];
};

vertex Corner cover_vertex(uint vid [[vertex_id]]) {
    float2 uv = float2(float((vid << 1) & 2u), float(vid & 2u));
    Corner out;
    out.position = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
    return out;
}

struct Cell {
    half4 c [[raster_order_group(0)]];
    ushort id [[raster_order_group(0)]];
};

struct MaskedOut {
    Cell cell [[imageblock_data]];
    half4 color [[color(0)]];
    uint mask [[sample_mask]];
};

static half4 pixel_color(float4 position) {
    return half4(half(uint(position.x)), half(uint(position.y)), 0.5h, 1.0h);
}

fragment MaskedOut cover_masked(float4 position [[position]]) {
    uint x = uint(position.x);
    uint y = uint(position.y);
    MaskedOut out;
    out.cell.c = half4(half(x), half(y), 0.25h, -1.0h);
    out.cell.id = ushort(4u * y + x + 100u);
    out.color = pixel_color(position);
    uint k = (x + 2u * y) & 3u;
    out.mask = k == 0u ? 0u : k == 1u ? 1u : k == 2u ? 0xfffffffeu : 0xffffffffu;
    return out;
}

struct Wide {
    half a [[user(a), raster_order_group(0)]];
    half b [[user(b), raster_order_group(0)]];
    uint w [[user(w), raster_order_group(0)]];
};

struct Swapped {
    half b [[user(b), raster_order_group(0)]];
    half a [[user(a), raster_order_group(0)]];
};

struct Word {
    uint w [[user(w), raster_order_group(0)]];
};

struct ViewA {
    half a [[user(a)]];
};

struct ViewW {
    uint w [[user(w)]];
};

struct ViewAW {
    half a [[user(a)]];
    uint w [[user(w)]];
};

struct SwappedOut {
    ViewA view [[imageblock_data(Swapped)]];
    half4 color [[color(0)]];
};

struct WideOut {
    ViewA view [[imageblock_data(Wide)]];
    half4 color [[color(0)]];
};

struct WideAWOut {
    ViewAW view [[imageblock_data(Wide)]];
    half4 color [[color(0)]];
};

fragment SwappedOut cover_two_down(float4 position [[position]],
                                   ViewA in [[imageblock_data(Wide)]]) {
    SwappedOut out;
    out.view.a = in.a + 100.0h;
    out.color = pixel_color(position);
    return out;
}

fragment WideOut cover_two_up(float4 position [[position]],
                              ViewA in [[imageblock_data(Swapped)]]) {
    WideOut out;
    out.view.a = in.a + 1000.0h;
    out.color = pixel_color(position);
    return out;
}

fragment WideAWOut cover_two_word(float4 position [[position]],
                                  ViewW in [[imageblock_data(Word)]]) {
    WideAWOut out;
    out.view.a = as_type<half>(ushort(in.w >> 16));
    out.view.w = in.w & 0xffffu;
    out.color = pixel_color(position);
    return out;
}
