// Owned fixture (written for this repository): a two-plane
// extended-range texture held as two single-plane textures, as a consumer holds Apple's
// `RGB10A8_2P_XR10`: plane 0 viewed in `BGR10_XR` (or its sRGB twin) and the alpha plane in a
// four-byte format whose first component is the texel's second byte (`ARGB8Unorm`). One thread
// per texel, `half` reads and writes, no sampling, and the grid is the textures' size. Loop-free.
#include <metal_stdlib>
using namespace metal;

// Plane 0's color and the alpha view's first component, joined into one `BGRA10_XR` texture.
kernel void xr_planes_join(texture2d<half, access::read> color [[texture(0)]],
                           texture2d<half, access::read> alpha [[texture(1)]],
                           texture2d<half, access::write> joined [[texture(2)]],
                           uint2 gid [[thread_position_in_grid]])
{
    half4 c = color.read(gid);
    half a = alpha.read(gid).r;
    joined.write(half4(c.rgb, a), gid);
}

// One `BGRA10_XR` texture split into plane 0, written with alpha 1, and the alpha view, written
// with the alpha in every component.
kernel void xr_planes_split(texture2d<half, access::read> joined [[texture(0)]],
                            texture2d<half, access::write> color [[texture(1)]],
                            texture2d<half, access::write> alpha [[texture(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    half4 v = joined.read(gid);
    color.write(half4(v.rgb, 1.0h), gid);
    alpha.write(half4(v.a), gid);
}
