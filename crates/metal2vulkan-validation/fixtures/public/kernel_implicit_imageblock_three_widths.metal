// `air.load.implicit_imageblock.{f32,f16,i32}` and `air.store.implicit_imageblock.f32` had no
// authored case. One implicit imageblock with a `float`, a `half` and a `uint` member reaches every
// width of both families at once, because an implicit-layout imageblock member IS its colour
// attachment and the three attachments can carry three different formats.
//
// The integer member is `uint` rather than `int` on purpose. AIR mangles both as `i32` -- the
// intrinsic is `air.load.implicit_imageblock.i32` either way -- and the attachment format is read
// off that suffix, so reflection declares `R32Uint` for both. The only corpus source that reaches
// this family declares a `uint` member too, so `uint` is the shape worth pinning; a signed member
// would need the render-target type name out of the imageblock layout metadata, which nothing reads
// today and nothing in the corpus needs.
//
// The existing `kernel_implicit_imageblock_half2` fixture copies the block to itself and its case
// observes a single 1x1 region, so it says nothing about the imageblock COORDINATE. This one has
// every thread write its own row of the output buffer, keyed by `thread_position_in_threadgroup`,
// and every cell of the three attachments carries its own linear index -- so a swapped or dropped
// coordinate is visible in all 256 cells of all three planes rather than in none.
//
// The three write-backs are deliberately different arithmetic per width (`+1.0f`, `*2.0h`, `+100`)
// so that a store that landed in the wrong plane could not be mistaken for a correct one.
//
// A tile dispatch is size-locked: the imageblock dimensions must equal the threadgroup and the grid
// must be exactly one threadgroup, or several tiles copy their blocks to overlapping origins. 16x16
// is the shape Metal accepts on this device.
#include <metal_stdlib>
using namespace metal;

struct Block {
    float  scale [[color(0)]];
    half   weight [[color(1)]];
    uint   tag [[color(2)]];
};

kernel void kernel_implicit_imageblock_three_widths(
    imageblock<Block, imageblock_layout_implicit> blk,
    device uint* out [[buffer(0)]],
    ushort2 pos [[thread_position_in_threadgroup]])
{
    Block v = blk.read(pos);

    // Each thread writes its OWN row, so the case pins the imageblock coordinate for all 256
    // cells rather than for one.
    uint row = (uint(pos.y) * 16u + uint(pos.x)) * 3u;
    out[row + 0] = as_type<uint>(v.scale);
    out[row + 1] = uint(as_type<ushort>(v.weight));
    out[row + 2] = as_type<uint>(v.tag);

    v.scale = v.scale + 1.0f;
    v.weight = v.weight * 2.0h;
    v.tag = v.tag + 100;
    blk.write(v, pos);
}
