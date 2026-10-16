// Owned fixture: framebuffer fetch (`[[color(n)]]` fragment inputs)
// through input attachments. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Blended {
    float4 color [[color(0)]];
    uint4 ids [[color(1)]];
};

// Programmable blending: a float, an unsigned, and a half attachment read, two written back,
// with a constant blend factor.
fragment Blended fetch_blend(float4 position [[position]],
                             float4 dst [[color(0)]],
                             uint4 ids [[color(1)]],
                             half2 extra [[color(2)]],
                             constant float4 &factor [[buffer(0)]]) {
    Blended out;
    out.color = dst * factor + float4(float2(extra), position.x, 0.25);
    out.ids = ids + uint4(1u, 2u, 3u, 4u);
    return out;
}

// A single-channel attachment read as a scalar.
fragment float fetch_scalar(float dst [[color(0)]]) {
    return dst * 0.5;
}

// A color input in a raster order group: read inside the fragment interlock.
fragment float4 fetch_ordered(float4 dst [[color(0), raster_order_group(0)]]) {
    return dst * 0.5;
}
