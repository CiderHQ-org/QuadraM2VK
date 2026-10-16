// Owned fixture: a raster order group on a color output
// the function does not read, drawn over overlapping triangles whose blend does not commute, so
// the target holds the answer of one order alone. Written for this repository; no third-party
// source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct RogLayer {
    float4 position [[position]];
    float value [[flat]];
};

// Triangle `k` (vertices 3k to 3k + 2) is Metal's fullscreen triangle, (-1, -1), (3, -1),
// (-1, 3), carrying the value (k + 1) / 8.
vertex RogLayer rog_order_layers(uint vid [[vertex_id]]) {
    uint c = vid % 3u;
    float2 a = float2(float(c & 1u), float(c >> 1));
    RogLayer out;
    out.position = float4(4.0 * a - 1.0, 0.5, 1.0);
    out.value = float(vid / 3u + 1u) * 0.125;
    return out;
}

struct RogOrdered {
    float4 color [[color(0), raster_order_group(0)]];
};

// A color output in a group that the function does not read: (v, 1 - v, v / 2, 1 / 2).
fragment RogOrdered rog_order_output(RogLayer in [[stage_in]]) {
    RogOrdered out;
    out.color = float4(in.value, 1.0 - in.value, 0.5 * in.value, 0.5);
    return out;
}
