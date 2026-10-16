// Owned fixture: a raster order group on an argument-buffer member.
// The same read-modify-write as a3_rog_buffer_rmw (v -> 3v + 1 on the word of the fragment's
// pixel, 8 words per row), through a device pointer held in an argument buffer whose member
// carries raster_order_group(0). Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Targets {
    device uint *acc [[id(0), raster_order_group(0)]];
};

fragment float4 a3_rog_argument_buffer(float4 pos [[position]],
                                       constant Targets &targets [[buffer(0)]]) {
    uint i = uint(pos.y) * 8u + uint(pos.x);
    uint v = targets.acc[i];
    targets.acc[i] = v * 3u + 1u;
    return float4(0.25f, 0.5f, 0.75f, 1.0f);
}
