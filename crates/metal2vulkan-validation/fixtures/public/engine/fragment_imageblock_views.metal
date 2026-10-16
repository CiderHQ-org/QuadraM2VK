// Owned fixture: fragment imageblocks through the imageblock storage.
//
// `imageblock_transition` reads one view of a master layout and writes another: the input view
// holds `b` (a half at byte 2, the high half of word 0) and `sum` (a float at byte 8), the output
// view `a` (a half at byte 0, the low half of word 0), `tag` (a uchar4 at byte 4), and `n` (a
// ushort at byte 12). Every member of a view lives at its master offset, so the output's `a`
// shares a word with the input's `b`, which must survive the write.
//
// `imageblock_accumulate` reads and writes its whole layout (no master) after the early fragment
// tests; `imageblock_clear_odd` writes it only (a discarded fragment writes nothing), and
// `imageblock_resolve` reads it only (and scales it by a buffer's value). `imageblock_masked`
// writes it and returns a sample mask, which can drop the fragment after the write.
// `imageblock_two_masters` reads `a` through `Master` (byte 0) and writes it through `Swapped`
// (byte 2), `imageblock_aliased` writes an imageblock aliasing the implicit one (the color
// attachments' storage), and `imageblock_aliased_in` reads one. `plain_color` reaches no
// imageblock and no raster order group, although other layouts of the module name one.
// `imageblock_vertex` draws a full-viewport triangle for render pipelines of these functions.
#include <metal_stdlib>
using namespace metal;

struct Corner {
    float4 position [[position]];
};

vertex Corner imageblock_vertex(uint vid [[vertex_id]]) {
    float2 uv = float2(float((vid << 1) & 2u), float(vid & 2u));
    Corner out;
    out.position = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
    return out;
}

struct Master {
    half a [[user(a), raster_order_group(0)]];
    half b [[user(b), raster_order_group(0)]];
    uchar4 tag [[user(tag), raster_order_group(0)]];
    float sum [[user(sum), raster_order_group(0)]];
    ushort n [[user(n), raster_order_group(0)]];
};

struct ViewIn {
    half b [[user(b)]];
    float sum [[user(sum)]];
};

struct ViewOut {
    half a [[user(a)]];
    uchar4 tag [[user(tag)]];
    ushort n [[user(n)]];
};

struct TransitionOut {
    half4 color [[color(0)]];
    ViewOut view [[imageblock_data(Master)]];
};

fragment TransitionOut imageblock_transition(ViewIn in [[imageblock_data(Master)]],
                                             float4 position [[position]]) {
    TransitionOut out;
    out.color = half4(in.b, half(in.sum), 0.0h, 1.0h);
    out.view.a = in.b + 1.0h;
    out.view.tag = uchar4(uint(position.x), uint(position.y), 7, 9);
    out.view.n = ushort(in.sum * 2.0f);
    return out;
}

struct Cell {
    half4 color [[raster_order_group(0)]];
    half depth [[raster_order_group(0)]];
};

struct CellOut {
    Cell cell [[imageblock_data]];
};

[[early_fragment_tests]] fragment CellOut imageblock_accumulate(Cell in [[imageblock_data]],
                                                         float4 position [[position]]) {
    CellOut out;
    out.cell.color = in.color * 0.5h + half4(half(position.x), half(position.y), 0.25h, 0.5h);
    out.cell.depth = in.depth + 1.0h;
    return out;
}

fragment CellOut imageblock_clear_odd(float4 position [[position]]) {
    if ((uint(position.x) & 1u) != 0u) {
        discard_fragment();
    }
    CellOut out;
    out.cell.color = half4(0.0h);
    out.cell.depth = half(position.y);
    return out;
}

fragment half4 imageblock_resolve(Cell in [[imageblock_data]],
                                  constant half &scale [[buffer(0)]]) {
    return in.color * in.depth * scale;
}

struct MaskedOut {
    Cell cell [[imageblock_data]];
    uint mask [[sample_mask]];
};

fragment MaskedOut imageblock_masked(float4 position [[position]]) {
    MaskedOut out;
    out.cell.color = half4(1.0h);
    out.cell.depth = half(position.x);
    out.mask = uint(position.y) & 1u;
    return out;
}

struct Swapped {
    half b [[user(b), raster_order_group(0)]];
    half a [[user(a), raster_order_group(0)]];
};

struct ViewA {
    half a [[user(a)]];
};

struct SwappedOut {
    ViewA view [[imageblock_data(Swapped)]];
};

fragment SwappedOut imageblock_two_masters(ViewA in [[imageblock_data(Master)]]) {
    SwappedOut out;
    out.view.a = in.a + 1.0h;
    return out;
}

struct Implicit {
    half4 color [[raster_order_group(0)]];
};

struct AliasedOut {
    Implicit i [[imageblock_data, alias_implicit_imageblock]];
};

fragment AliasedOut imageblock_aliased(float4 position [[position]]) {
    AliasedOut out;
    out.i.color = half4(half(position.x), half(position.y), 0.0h, 1.0h);
    return out;
}

fragment half4 imageblock_aliased_in(Implicit i [[imageblock_data, alias_implicit_imageblock]]) {
    return i.color * 0.5h;
}

fragment half4 plain_color(float4 position [[position]]) {
    return half4(half(position.x), half(position.y), 0.0h, 1.0h);
}
