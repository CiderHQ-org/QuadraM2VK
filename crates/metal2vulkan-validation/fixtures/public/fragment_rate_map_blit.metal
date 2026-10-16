// A blit into the physical image of a render pass with a rasterization-rate map, through the
// map's decoder: each fragment reads the screen-space texel its pixel stands for.
//
// A case draws the generated fullscreen triangle through this function in a pass
// whose `rasterizationRateMap` the case authors (`render_pass_rate_map`), with buffer 0 the same
// map's parameter data (a `rasterization_rate_map` literal) and texture 0 a screen-sized image.
// Each fragment maps its `[[position]]` (a physical pixel's center) to screen coordinates with
// `map_physical_to_screen_coordinates`, and writes the texel of `screen` there, clamped to the
// image: the pass's target holds the screen image warped into the map's physical layout.
#include <metal_stdlib>
using namespace metal;

fragment uint fragment_rate_map_blit(float4 position [[position]],
                                     constant rasterization_rate_map_data& map [[buffer(0)]],
                                     texture2d<uint, access::read> screen [[texture(0)]]) {
    rasterization_rate_map_decoder decoder(map);
    float2 point = floor(decoder.map_physical_to_screen_coordinates(position.xy));
    float2 last = float2(screen.get_width() - 1, screen.get_height() - 1);
    uint2 texel = uint2(clamp(point, float2(0.0f), last));
    return screen.read(texel).r;
}
