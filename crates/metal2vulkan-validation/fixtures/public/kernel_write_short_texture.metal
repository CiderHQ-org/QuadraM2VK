// A kernel whose only output is a texture2d<short, access::write>. The written components are
// chosen to lie outside the signed byte range, so an 8-bit signed storage image cannot carry them:
// the emitter decorated this texture `Rgba8i` until the signed storage default learned to widen for
// `<short` the way the unsigned one already widened for `<ushort`.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_write_short_texture(
    texture2d<short, access::write> out [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    out.write(short4(-32768, -129, 128, 32767), gid);
}
