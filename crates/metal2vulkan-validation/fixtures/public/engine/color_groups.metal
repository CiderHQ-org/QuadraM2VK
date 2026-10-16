// Owned fixture: render pipelines of more color attachments than a device at Vulkan's floor draws
// at once (four), which the translation draws in groups of four (`pipeline::compile::color_group`).
// A fullscreen triangle's varying `tint` feeds every color. `cg_five` writes colors 0 to 4,
// `cg_seven` 0 to 6, `cg_sparse` 0 and 5; `cg_depth` five colors, a depth and a sample mask, and
// discards; `cg_counts` five colors and counts itself in a device atomic. Metal 2.2, whose discard
// ends the invocation (`color_groups_fetch` reads a color). Loop-free.
// metal-std: macos-metal2.2
#include <metal_stdlib>
using namespace metal;

struct CgPoint {
    float4 position [[position]];
    float4 tint;
};

// The fullscreen triangle, (-1, -1), (3, -1), (-1, 3), computed rather than indexed from a local
// array (whose lifetime markers LLVM majors print differently).
vertex CgPoint cg_cover(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    CgPoint out;
    out.position = float4(4.0 * a - 1.0, 0.5 * a.x, 1.0);
    out.tint = float4(a, 1.0 - a.x, 0.25);
    return out;
}

struct CgFive {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
    float4 c4 [[color(4)]];
};

static CgFive five(float4 t) {
    return CgFive { t, t.yzwx, t.zwxy, t.wxyz, t * 0.5 };
}

fragment CgFive cg_five(CgPoint in [[stage_in]]) {
    return five(in.tint);
}

struct CgSeven {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
    float4 c4 [[color(4)]];
    float4 c5 [[color(5)]];
    float4 c6 [[color(6)]];
};

fragment CgSeven cg_seven(CgPoint in [[stage_in]]) {
    float4 t = in.tint;
    return CgSeven { t, t.yzwx, t.zwxy, t.wxyz, t * 0.5, t * 0.25, 1.0 - t };
}

struct CgSparse {
    float4 c0 [[color(0)]];
    float4 c5 [[color(5)]];
};

fragment CgSparse cg_sparse(CgPoint in [[stage_in]]) {
    return CgSparse { in.tint, in.tint.wzyx };
}

struct CgDepth {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
    float4 c4 [[color(4)]];
    float depth [[depth(any)]];
    uint mask [[sample_mask]];
};

fragment CgDepth cg_depth(CgPoint in [[stage_in]]) {
    if (in.tint.x > 0.75) {
        discard_fragment();
    }
    float4 t = in.tint;
    float depth = in.position.z * 0.5 + t.y * 0.25;
    uint mask = t.y > 0.5 ? 1u : 0xffffffffu;
    return CgDepth { t, t.yzwx, t.zwxy, t.wxyz, t * 0.5, depth, mask };
}

fragment CgFive cg_counts(CgPoint in [[stage_in]], device atomic_uint *count [[buffer(0)]]) {
    atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    return five(in.tint);
}
