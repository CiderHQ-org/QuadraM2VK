// Owned fixture: stage outputs and arguments a function constant gates.
// With no constant supplied every gate is off (MSL 5.8.1.2: an undefined constant behaves as if
// `is_function_constant_defined` were false), so each gated output and argument is absent:
// - a gated varying and a gated point size leave the vertex interface (MSL 5.8.1.6);
// - a gated color output leaves the fragment interface;
// - a gated buffer that the body reads without a guard reads as null (a read through the null
//   buffer pointer is the access family's null root, zero); a gated texture's observed read is
//   refused (`FunctionConstantArgument`: its value is undefined); a present
//   texture sampled through a gated sampler is refused: `Lowering(Sampler)`;
// - a gated color input whose read only feeds a blend the frontend selects away is never observed;
// - a gated input that only feeds an output gated by the same constant is never observed (the
//   output leaves the interface, so the return does not observe the field);
// - a gated sampler merged with a present one before a sample is refused
//   (`FunctionConstantArgument`): only a sampler read directly by a sample reads as null.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool has_tint [[function_constant(0)]];
constant bool has_size [[function_constant(1)]];
constant bool has_second [[function_constant(2)]];
constant bool has_source [[function_constant(3)]];
constant bool has_blend [[function_constant(4)]];
constant bool blend_defined = is_function_constant_defined(has_blend);
constant bool blend = blend_defined && has_blend;
constant bool has_uv [[function_constant(5)]];

struct GatedVaryings {
    float4 position [[position]];
    float4 tint [[function_constant(has_tint)]];
    float point_size [[point_size, function_constant(has_size)]];
    float4 shade;
};

// Two gated outputs between two ungated ones.
vertex GatedVaryings fc_vertex_gated_outputs(uint vid [[vertex_id]]) {
    GatedVaryings out;
    out.position = float4(float(vid & 1u), float(vid >> 1), 0.0, 1.0);
    out.tint = float4(0.25);
    out.point_size = 4.0;
    out.shade = float4(float(vid), 0.5, 0.0, 1.0);
    return out;
}

struct GatedTargets {
    float4 first [[color(0)]];
    float4 second [[color(1), function_constant(has_second)]];
};

// A gated second color output.
fragment GatedTargets fc_fragment_gated_color(float4 pos [[position]]) {
    GatedTargets out;
    out.first = float4(pos.xy, 0.0, 1.0);
    out.second = float4(1.0, 0.0, 0.0, 1.0);
    return out;
}

// A read and a sample of a gated texture and sampler, neither guarded: Metal leaves both
// undefined, so the translation refuses the read (`FunctionConstantArgument`).
kernel void fc_kernel_absent_resources(
    texture2d<float, access::read> source [[texture(0), function_constant(has_source)]],
    texture2d<float, access::sample> sampled [[texture(1), function_constant(has_source)]],
    sampler smp [[sampler(0), function_constant(has_source)]],
    texture2d<float, access::write> dest [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  float4 a = source.read(gid);
  float4 b = sampled.sample(smp, float2(0.5));
  dest.write(a + b + float4(1.0), gid);
}

// A present texture sampled through a gated sampler: Metal defines no result, so it is refused
// (`Lowering(Sampler)`) until measured.
kernel void fc_kernel_absent_sampler(
    texture2d<float, access::sample> sampled [[texture(0)]],
    sampler smp [[sampler(0), function_constant(has_source)]],
    texture2d<float, access::write> dest [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]) {
  dest.write(sampled.sample(smp, float2(0.5)), gid);
}

// An unguarded read of a gated buffer: the null pointer, which the null root reads as zero.
kernel void fc_kernel_absent_buffer(device const float *gone [[buffer(0), function_constant(has_source)]],
                                    device float *out [[buffer(1)]],
                                    uint gid [[thread_position_in_grid]]) {
  out[gid] = gone[gid] + 1.0;
}

// A gated color input that only feeds a blend the static-initialized `blend` flag selects away:
// with the gate off the read is never observed.
fragment float4 fc_fragment_unobserved_input(float4 pos [[position]],
                                             float4 prior [[color(0), function_constant(blend_defined)]]) {
  float4 base = float4(pos.xy * 0.5, 0.0, 1.0);
  float4 mixed = fma(prior, float4(0.5), base);
  return blend ? mixed : base;
}

struct GatedIn {
    float4 position [[attribute(0)]];
    float2 uv [[attribute(1), function_constant(has_uv)]];
};

struct GatedOut {
    float4 position [[position]];
    float2 uv [[function_constant(has_uv)]];
};

// A gated attribute that only feeds a varying gated by the same constant.
vertex GatedOut fc_vertex_input_to_output(GatedIn in [[stage_in]]) {
    GatedOut out;
    out.position = in.position;
    out.uv = in.uv;
    return out;
}

struct GatedColorOut {
    float4 blended [[color(0), function_constant(has_uv)]];
    float4 plain [[color(1)]];
};

// A gated color input that only feeds a color output gated by the same constant.
fragment GatedColorOut fc_fragment_input_to_output(float4 prior [[color(0), function_constant(has_uv)]],
                                                   float4 pos [[position]]) {
    GatedColorOut out;
    out.blended = prior * 0.5;
    out.plain = pos;
    return out;
}

// A gated sampler merged with a present one: the sample may go through the absent sampler.
kernel void fc_kernel_sampler_select(texture2d<float, access::sample> sampled [[texture(0)]],
                                     sampler here [[sampler(0)]],
                                     sampler gone [[sampler(1), function_constant(has_source)]],
                                     texture2d<float, access::write> dest [[texture(1)]],
                                     uint2 gid [[thread_position_in_grid]]) {
  sampler s = gid.x > 3 ? gone : here;
  dest.write(sampled.sample(s, float2(0.5), level(0)), gid);
}
