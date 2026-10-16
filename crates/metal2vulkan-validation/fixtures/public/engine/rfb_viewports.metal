// Owned MSL: what a fragment function reads of its position and facing, and the depth it writes,
// under the viewports a render encoder sets: depth ranges past [0, 1] in both depth clip modes,
// depth bias, and negative widths. The shaders of a Metal probe's viewport cases (an M3 Max and an
// M4), so each runtime lane draw asks the question the probe measured. Written from scratch for
// this repository; loop-free.
//
// rfb_full: Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth `z + slope * x`.
// rfb_position: the position the fragment reads (x, y, z) and its facing, +1 front, -1 back.
// rfb_depth: writes the depth `d`, and the position's z it read into red. rfb_span: a quad over y
// -1 to 1 from x0 to x1, its z running from z0 at x0 to z1 at x1 (w 1), drawn as a strip of four
// vertices: the shader of a Metal probe on an M3 Max and an M4, whose depth runs along x.
#include <metal_stdlib>
using namespace metal;

struct RfbOut {
    float4 position [[position]];
};

struct RfbDepth {
    float z;
    float slope;
};

vertex RfbOut rfb_full(uint vid [[vertex_id]], constant RfbDepth &depth [[buffer(0)]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    float2 p = 4.0 * a - 1.0;
    RfbOut out;
    out.position = float4(p, depth.z + depth.slope * p.x, 1.0);
    return out;
}

struct RfbSpan {
    float x0;
    float x1;
    float z0;
    float z1;
};

vertex RfbOut rfb_span(uint vid [[vertex_id]], constant RfbSpan &span [[buffer(0)]]) {
    bool right = (vid & 1u) != 0u;
    RfbOut out;
    out.position = float4(right ? span.x1 : span.x0, (vid >> 1) != 0u ? 1.0 : -1.0,
                          right ? span.z1 : span.z0, 1.0);
    return out;
}

fragment float4 rfb_position(RfbOut in [[stage_in]], bool front [[front_facing]]) {
    return float4(in.position.x, in.position.y, in.position.z, front ? 1.0 : -1.0);
}

struct RfbWritten {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

fragment RfbWritten rfb_depth(RfbOut in [[stage_in]], constant float &d [[buffer(0)]]) {
    RfbWritten out;
    out.color = float4(in.position.z, 0.0, 0.0, 1.0);
    out.depth = d;
    return out;
}
