// metal-std: macos-metal2.2
// Owned fixture: a terminate-semantics discard (before MSL 2.3,
// compiled with the standard named on the first line) in a raster order group. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Surface {
    float4 position [[position]];
    float4 color;
};

// Nothing tangled follows the discard, so it terminates; inside the group's body that is
// `discarded <- true; return`, and the entry kills after the critical section.
fragment float4 rog_terminate(Surface in [[stage_in]],
                              device float *acc [[buffer(0), raster_order_group(0)]]) {
    uint i = uint(in.position.x);
    acc[i] = acc[i] + 1.0;
    if (in.color.a < 0.5) {
        discard_fragment();
    }
    return in.color;
}
