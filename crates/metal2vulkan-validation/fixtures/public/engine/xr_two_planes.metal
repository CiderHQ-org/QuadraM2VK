// Owned fixture (written for this repository): kernel writes
// and reads of a texture of Apple's two-plane extended-range format (`RGB10A8_2P_XR10` or its sRGB
// twin) as `texture2d<half>` and `texture2d<float>`, one thread per texel of a grid row `w` texels
// wide, as a consumer runs them (threadgroups of 32 x 32, threads past the texture included).
// Loop-free.
#include <metal_stdlib>
using namespace metal;

// Writes `in[g.y * w + g.x]` to texel `g`.
kernel void xr2p_write_half(texture2d<half, access::write> t [[texture(0)]],
                            device const half4 *in [[buffer(0)]], constant uint &w [[buffer(1)]],
                            uint2 g [[thread_position_in_grid]])
{
    t.write(in[g.y * w + g.x], g);
}

kernel void xr2p_write_float(texture2d<float, access::write> t [[texture(0)]],
                             device const float4 *in [[buffer(0)]], constant uint &w [[buffer(1)]],
                             uint2 g [[thread_position_in_grid]])
{
    t.write(in[g.y * w + g.x], g);
}

// Reads texel `g` into `out[g.y * w + g.x]`.
kernel void xr2p_read_half(texture2d<half, access::read> t [[texture(0)]],
                           device half4 *out [[buffer(0)]], constant uint &w [[buffer(1)]],
                           uint2 g [[thread_position_in_grid]])
{
    out[g.y * w + g.x] = t.read(g);
}

kernel void xr2p_read_float(texture2d<float, access::read> t [[texture(0)]],
                            device float4 *out [[buffer(0)]], constant uint &w [[buffer(1)]],
                            uint2 g [[thread_position_in_grid]])
{
    out[g.y * w + g.x] = t.read(g);
}
