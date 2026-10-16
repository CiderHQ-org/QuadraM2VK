// Owned fixture: how Metal converts texels to `half` when a texture
// declared `access::read` or `access::read_write` reads them (`bcx_half_read` reads textures
// declared `access::sample`). Four read-only 2D `half` textures (the case binds a 32-bit float, a
// 16-bit unorm, an 8-bit unorm, and an sRGB format) and two read-write ones (a 32-bit float and
// an 8-bit unorm format) are each read at texel `(i mod 16, i div 16)` of lane `i`, which writes
// the six results, widened to `float`, to `out[6i .. 6i + 5]` in slot order. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void hlf_half_access(texture2d<half, access::read> r0 [[texture(0)]],
                            texture2d<half, access::read> r1 [[texture(1)]],
                            texture2d<half, access::read> r2 [[texture(2)]],
                            texture2d<half, access::read> r3 [[texture(3)]],
                            texture2d<half, access::read_write> w4 [[texture(4)]],
                            texture2d<half, access::read_write> w5 [[texture(5)]],
                            device float4 *out [[buffer(0)]],
                            uint tid [[thread_position_in_grid]])
{
    uint2 texel = uint2(tid % 16, tid / 16);
    device float4 *o = out + 6 * tid;
    o[0] = float4(r0.read(texel));
    o[1] = float4(r1.read(texel));
    o[2] = float4(r2.read(texel));
    o[3] = float4(r3.read(texel));
    o[4] = float4(w4.read(texel));
    o[5] = float4(w5.read(texel));
}
