// Owned fixture: fragment functions of a multisampled render pipeline.
// Metal reads `[[position]]` and center and centroid varyings at the pixel center in every
// invocation, and runs a function that writes memory once per covered sample; these entries read
// each kind of input beside the sample index, a sample-qualified varying, or a memory write.
// Loop-free; nothing here runs on a GPU.
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float2 uv;
    half2 shade;
    float tone [[center_no_perspective]];
    float4 sampled [[sample_perspective]];
    float4 edge [[centroid_perspective]];
};

// A strip of two triangles over the viewport, with a varying of each interpolation.
vertex Varyings multisample_vertex(uint vid [[vertex_id]]) {
    float2 p = float2(float(vid & 1u), float(vid >> 1));
    Varyings out;
    out.position = float4(p * 2.0 - 1.0, 0.5, 1.0);
    out.uv = p;
    out.shade = half2(p.yx);
    out.tone = p.x * 0.25;
    out.sampled = float4(p, 0.0, 1.0);
    out.edge = float4(p.yx, 1.0, 0.0);
    return out;
}

// Sample rate (`[[sample_id]]`) beside `[[position]]`'s x and y and three center varyings, each
// read at the pixel center.
fragment float4 multisample_center_at_sample(Varyings in [[stage_in]],
                                             uint sample [[sample_id]]) {
    return float4(in.position.xy + in.uv, float(in.shade.x) + in.tone, float(sample));
}

// Sample rate beside `[[position]]`'s z and w, which Metal also reads at the pixel center.
fragment float4 multisample_depth_at_sample(Varyings in [[stage_in]],
                                            uint sample [[sample_id]]) {
    return float4(in.position.z, in.position.w, float(sample), 1.0);
}

// A sample-qualified varying beside a center one.
fragment float4 multisample_sampled_varying(Varyings in [[stage_in]]) {
    return in.sampled + float4(in.uv, 0.0, 0.0);
}

// A centroid varying at sample rate.
fragment float4 multisample_centroid_at_sample(Varyings in [[stage_in]],
                                               uint sample [[sample_id]]) {
    return in.edge + float4(float(sample));
}

// A memory write without `[[sample_id]]`, beside `[[position]]` and a center varying.
fragment float4 multisample_counter(Varyings in [[stage_in]],
                                    device atomic_uint *count [[buffer(0)]]) {
    atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    return float4(in.uv, in.position.x, 1.0);
}
