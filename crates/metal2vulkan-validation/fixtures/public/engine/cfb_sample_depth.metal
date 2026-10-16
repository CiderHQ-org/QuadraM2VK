// Owned fixture: the functions a Metal probe's passes drew with (custom sample positions with a
// depth or stencil attachment, and the store action `MTLStoreActionCustomSampleDepthStore`), so
// each pass of the runtime's lane asks the question the probe measured: each vertex where buffer 0
// holds it, its depth included, so a pass stores a plane of depths whose value at each sample tells
// where the sample lies; a fragment function that writes white, and one that writes nothing.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

vertex float4 cfb_depth_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]]) {
    return positions[vid];
}

fragment float4 cfb_depth_white() {
    return float4(1.0);
}

fragment void cfb_depth_none() {
}
