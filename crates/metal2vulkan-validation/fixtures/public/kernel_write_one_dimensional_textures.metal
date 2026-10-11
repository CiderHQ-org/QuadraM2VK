// A kernel whose two destinations are the only image shapes whose SPIR-V capability the `Dim`
// enumerant does not settle. `Dim 1D` is enabled by either `Sampled1D` or `Image1D` and `Dim Buffer`
// by either `SampledBuffer` or `ImageBuffer`; which one describes the image is decided by the type's
// `Sampled` operand, and a rule that read that operand for the storage arm but not for the sampled
// one claimed both. Both textures here are `access::write`, so the module must declare `Image1D` and
// `ImageBuffer` and neither sampled capability -- there is no sampled image of either shape in it
// for one to name.
//
// `air.write_texture_buffer_1d.u.v4i32` appears in no corpus source at all, so no corpus host could
// ever have covered it. Every texel of both textures gets four distinct byte values derived from its
// own index, and the two destinations use different arithmetic, so a write that lands on the wrong
// texture, the wrong texel, or in the wrong lane order is a content mismatch rather than a silently
// identical image.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_write_one_dimensional_textures(
    texture1d<uint, access::write> line [[texture(0)]],
    texture_buffer<uint, access::write> texels [[texture(1)]],
    uint gid [[thread_position_in_grid]])
{
    line.write(uint4(0x10u + gid * 3u,
                     0xF0u - gid * 5u,
                     0x41u + gid * 7u,
                     0x90u ^ gid),
               gid);
    texels.write(uint4(0x20u + gid * 5u,
                       0xE1u - gid * 3u,
                       0x52u + gid * 11u,
                       0x70u ^ (gid * 2u)),
                 gid);
}
