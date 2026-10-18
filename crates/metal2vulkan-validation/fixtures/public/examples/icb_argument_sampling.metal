// Owned fixture: a render state that supports indirect command
// buffers and samples a texture through a constexpr sampler. Metal refuses such a state whose
// functions use a direct texture or sampler argument, so the fragment
// function reaches its texture through an argument buffer, as Metal allows. Written from scratch
// for the runtime lane; `scripts/regen-air-fixtures/regen-air-fixtures.sh --dir examples
// icb_argument_sampling` compiles it. No translator sweep reads this directory.
#include <metal_stdlib>
using namespace metal;

struct Varyings {
    float4 position [[position]];
    float4 color [[flat]];
};

// colored_vertex of runtime_examples.metal: positions and colors from two buffers by vertex ID.
vertex Varyings colored_vertex(device const float2 *positions [[buffer(0)]],
                               device const float4 *colors [[buffer(1)]],
                               uint id [[vertex_id]]) {
    Varyings out;
    out.position = float4(positions[id], 0.0, 1.0);
    out.color = colors[id];
    return out;
}

// The texture the fragment function samples, at field 0 of its argument buffer.
struct SampledArguments {
    texture2d<half> source [[id(0)]];
};

// Each pixel sampled from the argument buffer's half texture through a linear constexpr sampler
// (clamp-to-edge, Metal's default), at its position over an 8-texel side.
fragment half4 argument_sampled_fragment(Varyings in [[stage_in]],
                                         constant SampledArguments &arguments [[buffer(0)]]) {
    constexpr sampler linear(filter::linear);
    return arguments.source.sample(linear, in.position.xy / 8.0);
}
