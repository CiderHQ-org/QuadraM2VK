// Owned fixture: plain writers of color 0 (no color input) that a render pass instance can move
// into the interlocked storage form (an interlocked color its other draws fetch, or an
// extended-range color's `half` shadow), one per way the depth and stencil tests bear on a store
// made before them, and the fullscreen triangle they are drawn with. Each writer returns `tint`
// (buffer 0) to color 0; the discarding ones discard the fragments of every fourth column (`x` in
// `[4k, 4k + 1)`). Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct SmwVertex {
    float4 position [[position]];
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth `z`.
vertex SmwVertex smw_full(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    SmwVertex out;
    out.position = float4(4.0 * a - 1.0, z, 1.0);
    return out;
}

// No discard, no other write: a store before the tests stays Metal's under early tests.
fragment half4 smw_plain(constant half4 &tint [[buffer(0)]]) {
    return tint;
}

// Only its discard keeps it from early tests.
fragment half4 smw_discard(float4 p [[position]], constant half4 &tint [[buffer(0)]]) {
    if ((uint(p.x) & 3u) == 0u) {
        discard_fragment();
    }
    return tint;
}

// It discards and writes other memory: Metal runs that write for the fragments its tests drop.
fragment half4 smw_store(float4 p [[position]], constant half4 &tint [[buffer(0)]],
                         device atomic_uint *runs [[buffer(1)]]) {
    if ((uint(p.x) & 3u) == 0u) {
        discard_fragment();
    }
    atomic_fetch_add_explicit(runs, 1u, memory_order_relaxed);
    return tint;
}

struct SmwDepth {
    half4 color [[color(0)]];
    float depth [[depth(any)]];
};

// It returns its depth, which Metal tests after it.
fragment SmwDepth smw_depth(float4 p [[position]], constant half4 &tint [[buffer(0)]],
                            constant float &depth [[buffer(1)]]) {
    SmwDepth out;
    out.color = tint;
    out.depth = depth;
    return out;
}

struct SmwMask {
    half4 color [[color(0)]];
    uint mask [[sample_mask]];
};

// It returns a sample mask, which keeps samples before the tests.
fragment SmwMask smw_mask(constant half4 &tint [[buffer(0)]], constant uint &mask [[buffer(1)]]) {
    SmwMask out;
    out.color = tint;
    out.mask = mask;
    return out;
}

// It declares early fragment tests: Metal runs it only for the fragments that pass them.
[[early_fragment_tests]]
fragment half4 smw_early(float4 p [[position]], constant half4 &tint [[buffer(0)]],
                         device atomic_uint *runs [[buffer(1)]]) {
    atomic_fetch_add_explicit(runs, 1u, memory_order_relaxed);
    return tint;
}
