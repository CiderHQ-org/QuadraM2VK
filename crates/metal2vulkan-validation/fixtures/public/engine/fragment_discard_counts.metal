// metal-std: macos-metal2.2
// Owned fixture: simdgroup counts around an MSL 2.2
// discard, as differences that hold whatever the simdgroup's size and however quads pack into it.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct DiscardCountsVertex {
    float4 position [[position]];
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3).
vertex DiscardCountsVertex discard_counts_full(uint vid [[vertex_id]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    DiscardCountsVertex out;
    out.position = float4(4.0 * a - 1.0, 0.5, 1.0);
    return out;
}

// `whole` 0 discards each quad's right column, so every quad keeps a live thread; 1 discards the
// quads of a checkerboard of 2x2 quads whole. A kept thread writes (after - before, after - kept,
// before, after): the simdgroup's lane counts before and after the discard, and `kept`, after it,
// the count of lanes whose quad keeps a thread.
fragment int4 discard_counts(DiscardCountsVertex in [[stage_in]],
                             constant uint &whole [[buffer(0)]]) {
    uint2 p = uint2(in.position.xy);
    int before = simd_sum(1);
    bool quad_kept = whole == 0u || (((p.x >> 1) + (p.y >> 1)) & 1u) == 0u;
    bool thread_kept = whole == 0u ? (p.x & 1u) == 0u : quad_kept;
    if (!thread_kept) {
        discard_fragment();
    }
    int after = simd_sum(1);
    int kept = simd_sum(quad_kept ? 1 : 0);
    return int4(after - before, after - kept, before, after);
}
