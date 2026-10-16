// Owned fixture: primitive restart of indexed draws, the questions a Metal probe asked on an M4.
// Each vertex takes its position from buffer 0 at its vertex id, clamped to 65535, so every index
// past 65535 lands where vertex 65535 is; the fragment function adds one 8-bit unit through the
// pipeline's additive blend, so each pixel of the target counts the primitives that cover it.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6RestartOut {
    float4 position [[position]];
    float size [[point_size]];
};

vertex Rt6RestartOut rt6_restart_vertex(uint vid [[vertex_id]],
                                        const device float2 *positions [[buffer(0)]]) {
    Rt6RestartOut out;
    out.position = float4(positions[min(vid, 65535u)], 0.25, 1.0);
    out.size = 1.0;
    return out;
}

fragment float4 rt6_restart_count() {
    return float4(1.0 / 255.0, 0.0, 0.0, 1.0);
}
