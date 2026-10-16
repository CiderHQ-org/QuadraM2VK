// Owned fixture: framebuffer fetch of a blended 16-bit float attachment through an output of one
// to four lanes. Each fragment function reads color attachment 1 ([[color(1)]]) in its output's
// type and writes it through its first output, beside color attachments 0 and 2 (a `half4` each,
// which the pipeline gives no channel to write), under the compiler's default fast math: the shape
// of a macOS 26 window-server pass, whose pipeline blends color 1 (RG16Float) with the destination
// color times the source plus zero for color and alpha, under a mask of red and green. On a device
// that orders fetch through the pixel interlock alone the write goes to a storage image, which no
// Vulkan blend applies to, so the function computes Metal's blend itself. Metal blends an output of
// fewer than four lanes as four whose missing lanes, alpha among them, are zero.
//
// - fbl_narrow_half2, fbl_narrow_float2: the guest's shape, a `half2` or `float2` output that is
//   one of two loaded values, chosen by the fetched red: opaque to the blend Metal's compiler
//   appends.
// - fbl_narrow_half, fbl_narrow_half3, fbl_narrow_half4, fbl_narrow_float, fbl_narrow_float3,
//   fbl_narrow_float4: the same load through outputs of the other widths.
//
// fbl_narrow_vertex draws one fullscreen triangle. Loop-free.
// metal-macos-min: 27.0
#include <metal_stdlib>
using namespace metal;

struct FbnVertex {
    float4 position [[position]];
};

vertex FbnVertex fbl_narrow_vertex(uint vid [[vertex_id]])
{
    float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));
    FbnVertex o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    return o;
}

#define FBN_ENTRY(name, T, Z)                                                                  \
    struct name##_out {                                                                         \
        half4 first [[color(0)]];                                                               \
        T color [[color(1)]];                                                                   \
        half4 third [[color(2)]];                                                               \
    };                                                                                          \
    fragment name##_out name(FbnVertex in [[stage_in]], T dst [[color(1)]],                     \
                             const device T *src [[buffer(0)]])                                 \
    {                                                                                           \
        uint i = uint(in.position.y) * 64u + uint(in.position.x);                               \
        name##_out o;                                                                           \
        o.first = half4(0.25h);                                                                 \
        o.color = Z > 2.0h ? src[i + 4096u] : src[i];                                           \
        o.third = half4(0.5h);                                                                  \
        return o;                                                                               \
    }

FBN_ENTRY(fbl_narrow_half, half, half(dst))
FBN_ENTRY(fbl_narrow_half2, half2, half(dst.x))
FBN_ENTRY(fbl_narrow_half3, half3, half(dst.x))
FBN_ENTRY(fbl_narrow_half4, half4, half(dst.x))
FBN_ENTRY(fbl_narrow_float, float, half(dst))
FBN_ENTRY(fbl_narrow_float2, float2, half(dst.x))
FBN_ENTRY(fbl_narrow_float3, float3, half(dst.x))
FBN_ENTRY(fbl_narrow_float4, float4, half(dst.x))
