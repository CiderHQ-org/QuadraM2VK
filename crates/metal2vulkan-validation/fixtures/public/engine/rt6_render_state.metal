// Owned fixture: the functions the runtime's render lane draws its render state with, the shaders
// of a Metal probe on an M4 (macOS 27.0.1), so each lane draw asks the question the probe measured.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6Out {
    float4 position [[position]];
};

struct Rt6Depth {
    float z;
    float slope;
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth `z + slope * x`: counter-
// clockwise in Metal's y-up clip space.
vertex Rt6Out rt6_full(uint vid [[vertex_id]], constant Rt6Depth &depth [[buffer(0)]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    float2 p = 4.0 * a - 1.0;
    Rt6Out out;
    out.position = float4(p, depth.z + depth.slope * p.x, 1.0);
    return out;
}

// The same triangle with its first two vertices swapped: the other winding.
vertex Rt6Out rt6_flip(uint vid [[vertex_id]], constant Rt6Depth &depth [[buffer(0)]]) {
    uint v = vid == 0u ? 1u : (vid == 1u ? 0u : vid);
    float2 a = float2(float(v & 1u), float(v >> 1));
    float2 p = 4.0 * a - 1.0;
    Rt6Out out;
    out.position = float4(p, depth.z, 1.0);
    return out;
}

// A small triangle inside the target: (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5); a fourth vertex,
// (0.5, 0.5), makes a strip of two triangles over a quad.
vertex Rt6Out rt6_small(uint vid [[vertex_id]], constant Rt6Depth &depth [[buffer(0)]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    Rt6Out out;
    out.position = float4(a - 0.5, depth.z, 1.0);
    return out;
}

struct Rt6Point {
    float4 position [[position]];
    float size [[point_size]];
};

// One point of 4 pixels at the center.
vertex Rt6Point rt6_point(constant Rt6Depth &depth [[buffer(0)]]) {
    Rt6Point out;
    out.position = float4(0.0, 0.0, depth.z, 1.0);
    out.size = 4.0;
    return out;
}

fragment float4 rt6_white() {
    return float4(1.0);
}

fragment float4 rt6_half() {
    return float4(0.5, 0.5, 0.5, 0.5);
}
