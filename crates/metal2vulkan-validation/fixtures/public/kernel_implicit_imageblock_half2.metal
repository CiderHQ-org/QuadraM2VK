#include <metal_stdlib>
using namespace metal;

struct ImplicitColor {
    half2 value [[color(0)]];
};

kernel void kernel_implicit_imageblock_half2(
    imageblock<ImplicitColor, imageblock_layout_implicit> block,
    ushort2 position [[thread_position_in_threadgroup]]) {
    ImplicitColor color = block.read(position);
    block.write(color, position);
}
