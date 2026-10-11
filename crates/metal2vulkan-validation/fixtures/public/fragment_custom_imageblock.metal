#include <metal_stdlib>
using namespace metal;

struct CustomImageblock {
    half warped [[user(warped_type), raster_order_group(0)]];
    half depth [[user(depth), raster_order_group(0)]];
    half blending_weight [[user(blending_weight), raster_order_group(0)]];
    half depth_buffer [[user(depth_buffer), raster_order_group(0)]];
};

struct FragmentImageblockOutput {
    half4 color [[color(0)]];
    CustomImageblock imageblock [[imageblock_data]];
};

fragment FragmentImageblockOutput fragment_custom_imageblock(
    CustomImageblock input [[imageblock_data]]) {
    FragmentImageblockOutput output;
    output.color = half4(input.depth, 0.0h, 0.0h, 1.0h);
    output.imageblock = input;
    output.imageblock.depth += 1.0h;
    return output;
}
