// Owned fixture: framebuffer fetch of a blended 8-bit unorm attachment without fast math. Each
// fragment function reads color attachment 0 ([[color(0)]]), writes it through its first output
// and writes color attachment 1 (a `float4`, which the pipeline gives no channel to write), as
// fbl_fetch_blend8's do, but compiled without fast math: Metal's compiler then rounds the blend's
// product and its sum each. On a device that orders fetch through the pixel interlock alone the
// write goes to a storage image, which no Vulkan blend applies to, so the function computes
// Metal's blend itself.
//
// - fbl8_safe_loaded: a `half4` output that is one of two loaded colors, chosen by whether the
//   fetched alpha is zero.
// - fbl8_safe_loaded_float: the same as a `float4` output (reading the attachment as `float4`),
//   which Metal blends in `float`.
// - fbl8_safe_product: a `half4` output that is the loaded color times the fetched one.
//
// fbl8_safe_vertex draws one fullscreen triangle. Loop-free.
// metal-fast-math: off
// metal-macos-min: 27.0
#include <metal_stdlib>
using namespace metal;

struct Fbl8Vertex {
    float4 position [[position]];
};

vertex Fbl8Vertex fbl8_safe_vertex(uint vid [[vertex_id]])
{
    float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));
    Fbl8Vertex o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    return o;
}

struct Fbl8Out {
    half4 color [[color(0)]];
    float4 second [[color(1)]];
};

struct Fbl8OutFloat {
    float4 color [[color(0)]];
    float4 second [[color(1)]];
};

fragment Fbl8Out fbl8_safe_loaded(Fbl8Vertex in [[stage_in]], half4 dst [[color(0)]],
                                  const device half4 *src [[buffer(0)]])
{
    uint i = uint(in.position.y) * 64u + uint(in.position.x);
    Fbl8Out o;
    o.color = dst.a == 0.0h ? src[i + 4096u] : src[i];
    o.second = float4(0.25f);
    return o;
}

fragment Fbl8OutFloat fbl8_safe_loaded_float(Fbl8Vertex in [[stage_in]], float4 dst [[color(0)]],
                                             const device float4 *src [[buffer(0)]])
{
    uint i = uint(in.position.y) * 64u + uint(in.position.x);
    Fbl8OutFloat o;
    o.color = dst.a == 0.0f ? src[i + 4096u] : src[i];
    o.second = float4(0.25f);
    return o;
}

fragment Fbl8Out fbl8_safe_product(Fbl8Vertex in [[stage_in]], half4 dst [[color(0)]],
                                   const device half4 *src [[buffer(0)]])
{
    uint i = uint(in.position.y) * 64u + uint(in.position.x);
    Fbl8Out o;
    o.color = src[i] * dst;
    o.second = float4(0.25f);
    return o;
}
