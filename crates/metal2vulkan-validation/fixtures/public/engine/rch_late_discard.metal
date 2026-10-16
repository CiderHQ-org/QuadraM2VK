// Owned fixture: fragment functions that read color attachment 0
// (`[[color(0)]]`: the storage form inside the pixel interlock where the device orders fetch only
// through it), discard a fragment whose attachment alpha is zero (MSL's default standard: demote
// semantics), and sample a texture. `rch_late_discard` writes nothing but its color, so its
// discard can wait until after its stores; `rch_late_discard_store` also writes a buffer after
// the discard, and `rch_late_discard_store_unsampled` does that without sampling.
// `rch_late_discard_plain` reads no attachment (no interlock): it samples, discards a texel of
// alpha zero, and returns a color computed after the discard.
#include <metal_stdlib>
using namespace metal;

constexpr sampler noise_sampler(filter::linear, address::repeat);

fragment half4 rch_late_discard(float4 position [[position]],
                                half4 dest [[color(0)]],
                                texture2d<half> noise [[texture(0)]],
                                constant half4 &tint [[buffer(0)]])
{
    half4 n = noise.sample(noise_sampler, position.xy * 0.03125f);
    if (dest.a == 0.0h) {
        discard_fragment();
    }
    return mix(dest, tint * n, tint.a);
}

fragment half4 rch_late_discard_store(float4 position [[position]],
                                      half4 dest [[color(0)]],
                                      texture2d<half> noise [[texture(0)]],
                                      constant half4 &tint [[buffer(0)]],
                                      device uint *written [[buffer(1)]])
{
    half4 n = noise.sample(noise_sampler, position.xy * 0.03125f);
    if (dest.a == 0.0h) {
        discard_fragment();
    }
    written[uint(position.y) * 64u + uint(position.x)] = 1u;
    return mix(dest, tint * n, tint.a);
}

fragment half4 rch_late_discard_store_unsampled(float4 position [[position]],
                                                half4 dest [[color(0)]],
                                                constant half4 &tint [[buffer(0)]],
                                                device uint *written [[buffer(1)]])
{
    if (dest.a == 0.0h) {
        discard_fragment();
    }
    written[uint(position.y) * 64u + uint(position.x)] = 1u;
    return mix(dest, tint, tint.a);
}

fragment half4 rch_late_discard_plain(float4 position [[position]],
                                      texture2d<half> noise [[texture(0)]],
                                      constant half4 &tint [[buffer(0)]])
{
    half4 n = noise.sample(noise_sampler, position.xy * 0.03125f);
    if (n.a == 0.0h) {
        discard_fragment();
    }
    return n * tint + tint.a;
}
