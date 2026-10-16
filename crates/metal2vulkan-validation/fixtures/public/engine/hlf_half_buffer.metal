// Owned fixture: how Metal converts a texture buffer's texels to `half`
// (the 2D reads `hlf_half_access` reads are not texture buffers). Four `half` texture buffers
// (the case binds a 32-bit float, an 8-bit unorm, an 8-bit snorm, and a half-float format) are
// each read at element `i` of lane `i`, which writes the four results, widened to `float`, to
// `out[4i .. 4i + 3]` in slot order. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void hlf_half_buffer(texture_buffer<half, access::read> b0 [[texture(0)]],
                            texture_buffer<half, access::read> b1 [[texture(1)]],
                            texture_buffer<half, access::read> b2 [[texture(2)]],
                            texture_buffer<half, access::read> b3 [[texture(3)]],
                            device float4 *out [[buffer(0)]],
                            uint tid [[thread_position_in_grid]])
{
    device float4 *o = out + 4 * tid;
    o[0] = float4(b0.read(tid));
    o[1] = float4(b1.read(tid));
    o[2] = float4(b2.read(tid));
    o[3] = float4(b3.read(tid));
}
