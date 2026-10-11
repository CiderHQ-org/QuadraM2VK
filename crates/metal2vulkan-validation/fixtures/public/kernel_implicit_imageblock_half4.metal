#include <metal_stdlib>
using namespace metal;

struct ImplicitColor {
    half4 value [[color(0)]];
};

kernel void kernel_implicit_imageblock_half4(
    imageblock<ImplicitColor, imageblock_layout_implicit> block,
    device half4* observed [[buffer(0)]],
    ushort2 position [[thread_position_in_threadgroup]]) {
    ImplicitColor color = block.read(position);
    color.value += half4(1.0h);
    block.write(color, position);
    observed[position.y * 16 + position.x] = block.read(position).value;
}
