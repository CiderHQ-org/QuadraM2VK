// Owned fixture: a non-atomic read-modify-write of a device buffer in
// raster order group 0. Every fragment of every overlapping primitive applies v -> 3v + 1 to the
// word of its pixel (8 words per row); the group orders the overlapping fragments' accesses, so no
// update is lost. Loop-free.
#include <metal_stdlib>
using namespace metal;

fragment float4 a3_rog_buffer_rmw(float4 pos [[position]],
                                  device uint *acc [[buffer(0), raster_order_group(0)]]) {
    uint i = uint(pos.y) * 8u + uint(pos.x);
    uint v = acc[i];
    acc[i] = v * 3u + 1u;
    return float4(0.25f, 0.5f, 0.75f, 1.0f);
}
