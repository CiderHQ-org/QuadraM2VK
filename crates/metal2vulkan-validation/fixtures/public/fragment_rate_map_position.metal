// Where each fragment of a render pass with a rasterization-rate map lands and where its
// attributes are evaluated.
//
// A case draws the generated fullscreen triangle through this function in a pass whose
// `rasterizationRateMap` the case authors (`render_pass_rate_map`), into an `rgba32_float` target
// of the map's screen size. The passthrough vertex writes `uv = (x / W, 1 - y / H)` for the screen
// point `(x, y)` of each vertex, linear in screen space, so a fragment reads back the screen point
// its attributes were evaluated at. Each fragment writes its `[[position]]` (on Apple M3, the
// center of its pixel in the physical target, as our probe measured) and that screen point rounded
// down to whole pixels, which keeps the observation exact where the interpolation rounds in its
// last bits. `screen` is the map's screen size.
#include <metal_stdlib>
using namespace metal;

struct RateMapFragmentInput {
    float4 position [[position]];
    float2 uv;
};

fragment float4 fragment_rate_map_position(RateMapFragmentInput input [[stage_in]],
                                           constant float2& screen [[buffer(0)]]) {
    float2 point = float2(input.uv.x, 1.0f - input.uv.y) * screen;
    return float4(input.position.xy, floor(point));
}
