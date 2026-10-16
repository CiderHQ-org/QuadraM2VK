// Owned fixture: a fragment function of five colors that reads color 4 (`[[color(4)]]`), which a
// device at Vulkan's floor draws in two groups of colors (`pipeline::compile::color_group`), and
// one of colors 0 and 5 that reads color 5, which it draws in one group renumbered below four.
// The vertex function is `color_groups`' `cg_cover`. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct CgPoint {
    float4 position [[position]];
    float4 tint;
};

struct CgFive {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
    float4 c4 [[color(4)]];
};

fragment CgFive cg_fetch(CgPoint in [[stage_in]], float4 last [[color(4)]]) {
    float4 t = in.tint;
    return CgFive { t, t.yzwx, t.zwxy, t.wxyz, last * 0.5 + t };
}

struct CgSparse {
    float4 c0 [[color(0)]];
    float4 c5 [[color(5)]];
};

fragment CgSparse cg_fetch_sparse(CgPoint in [[stage_in]], float4 last [[color(5)]]) {
    return CgSparse { in.tint, last * 0.5 + in.tint.wzyx };
}
