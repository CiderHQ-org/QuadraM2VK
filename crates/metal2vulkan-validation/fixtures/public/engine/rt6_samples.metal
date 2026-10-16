// Owned fixture: the runtime's per-sample fill and readback of a
// multisampled texture, the Metal oracle's own (`metal::platform`: the multisample initializer of
// `textures.rs` and the per-sample readback of `sample_readback.rs`), with the extent and sample
// count read from a buffer where the oracle writes them into its source.
//
// - Fill: a render pass per slice draws the fullscreen triangle at sample rate, and each
//   invocation writes its own sample's value, `values[(y * width + x) * samples + sample]`, in
//   the format's value type, as the oracle's `metal2vulkan_literal_fragment` does: one fragment
//   function per format the oracle fills, and `[[depth(any)]]` for `Depth32Float`.
// - Readback: one thread per sample (threadgroups of one, the grid the pixels by the samples)
//   reads its sample and writes the texel's words at its place in the sample-minor layout, as the
//   oracle's `metal2vulkan_sample_readback` does: a `Depth32Float` texture as its bits, a
//   `Stencil8` one as its first component, and a color one through an unsigned view with its own
//   components (a device without Vulkan's `imageViewFormatReinterpretation` views no format as one
//   of other components), whose components are packed into the words the oracle's view of the
//   texel size gives (one, two or four).
// Loop-free.
#include <metal_stdlib>
using namespace metal;

// The extent's width and the samples of each pixel.
struct Rt6Shape {
    uint width;
    uint samples;
};

struct Rt6FillVertex {
    float4 position [[position]];
};

// The oracle's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), computed rather than indexed
// from a local array (whose lifetime markers LLVM majors print differently).
vertex Rt6FillVertex rt6_fill_vertex(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return Rt6FillVertex { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

// The value of this invocation's sample.
template <typename T>
T sample_value(Rt6FillVertex input, uint sample, const device T *values, constant Rt6Shape &shape) {
    uint x = uint(input.position.x);
    uint y = uint(input.position.y);
    return values[(y * shape.width + x) * shape.samples + sample];
}

fragment float rt6_fill_r8unorm(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device uchar *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    uchar value = sample_value(input, sample, values, shape);
    return float(value) / 255.0f;
}

fragment float4 rt6_fill_rgba8unorm(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device uchar4 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    uchar4 value = sample_value(input, sample, values, shape);
    return float4(value) / 255.0f;
}

fragment uint4 rt6_fill_rgba8uint(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device uchar4 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    uchar4 value = sample_value(input, sample, values, shape);
    return uint4(value);
}

fragment int4 rt6_fill_rgba8sint(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device char4 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    char4 value = sample_value(input, sample, values, shape);
    return int4(value);
}

fragment half rt6_fill_r16float(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device half *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    half value = sample_value(input, sample, values, shape);
    return value;
}

fragment uint rt6_fill_r16uint(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device ushort *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    ushort value = sample_value(input, sample, values, shape);
    return uint(value);
}

fragment half2 rt6_fill_rg16float(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device half2 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    half2 value = sample_value(input, sample, values, shape);
    return value;
}

fragment float2 rt6_fill_rg32float(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device float2 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    float2 value = sample_value(input, sample, values, shape);
    return value;
}

fragment half4 rt6_fill_rgba16float(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device half4 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    half4 value = sample_value(input, sample, values, shape);
    return value;
}

fragment uint4 rt6_fill_rgba16uint(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device ushort4 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    ushort4 value = sample_value(input, sample, values, shape);
    return uint4(value);
}

fragment int4 rt6_fill_rgba16sint(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device short4 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    short4 value = sample_value(input, sample, values, shape);
    return int4(value);
}

fragment uint rt6_fill_r32uint(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device uint *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    uint value = sample_value(input, sample, values, shape);
    return value;
}

fragment int rt6_fill_r32sint(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device int *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    int value = sample_value(input, sample, values, shape);
    return value;
}

fragment float rt6_fill_r32float(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device float *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    float value = sample_value(input, sample, values, shape);
    return value;
}

fragment uint4 rt6_fill_rgba32uint(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device uint4 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    uint4 value = sample_value(input, sample, values, shape);
    return value;
}

fragment int4 rt6_fill_rgba32sint(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device int4 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    int4 value = sample_value(input, sample, values, shape);
    return value;
}

fragment float4 rt6_fill_rgba32float(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device float4 *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    float4 value = sample_value(input, sample, values, shape);
    return value;
}

struct Rt6Depth {
    float depth [[depth(any)]];
};

fragment Rt6Depth rt6_fill_depth32float(Rt6FillVertex input [[stage_in]], uint sample [[sample_id]],
        const device float *values [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]]) {
    return Rt6Depth { sample_value(input, sample, values, shape) };
}

// Where a thread's sample's words start.
uint words_base(uint3 position, constant Rt6Shape &shape, uint words) {
    return ((position.y * shape.width + position.x) * shape.samples + position.z) * words;
}

kernel void rt6_read_words1(texture2d_ms<uint, access::read> source [[texture(0)]],
        device uint *words [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]],
        uint3 position [[threadgroup_position_in_grid]]) {
    uint4 texel = source.read(position.xy, position.z);
    uint base = words_base(position, shape, 1u);
    words[base] = texel[0];
}

kernel void rt6_read_words2(texture2d_ms<uint, access::read> source [[texture(0)]],
        device uint *words [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]],
        uint3 position [[threadgroup_position_in_grid]]) {
    uint4 texel = source.read(position.xy, position.z);
    uint base = words_base(position, shape, 2u);
    words[base] = texel[0];
    words[base + 1u] = texel[1];
}

kernel void rt6_read_words4(texture2d_ms<uint, access::read> source [[texture(0)]],
        device uint *words [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]],
        uint3 position [[threadgroup_position_in_grid]]) {
    uint4 texel = source.read(position.xy, position.z);
    uint base = words_base(position, shape, 4u);
    words[base] = texel[0];
    words[base + 1u] = texel[1];
    words[base + 2u] = texel[2];
    words[base + 3u] = texel[3];
}

// A view of four 8-bit components: the texel's four bytes, in order, in one word.
kernel void rt6_read_bytes4(texture2d_ms<uint, access::read> source [[texture(0)]],
        device uint *words [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]],
        uint3 position [[threadgroup_position_in_grid]]) {
    uint4 texel = source.read(position.xy, position.z) & 0xffu;
    words[words_base(position, shape, 1u)] =
        texel[0] | (texel[1] << 8) | (texel[2] << 16) | (texel[3] << 24);
}

// A view of two 16-bit components: the texel's two halves, in order, in one word.
kernel void rt6_read_halves2(texture2d_ms<uint, access::read> source [[texture(0)]],
        device uint *words [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]],
        uint3 position [[threadgroup_position_in_grid]]) {
    uint4 texel = source.read(position.xy, position.z) & 0xffffu;
    words[words_base(position, shape, 1u)] = texel[0] | (texel[1] << 16);
}

// A view of four 16-bit components: the texel's four halves, in order, in two words.
kernel void rt6_read_halves4(texture2d_ms<uint, access::read> source [[texture(0)]],
        device uint *words [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]],
        uint3 position [[threadgroup_position_in_grid]]) {
    uint4 texel = source.read(position.xy, position.z) & 0xffffu;
    uint base = words_base(position, shape, 2u);
    words[base] = texel[0] | (texel[1] << 16);
    words[base + 1u] = texel[2] | (texel[3] << 16);
}

kernel void rt6_read_depth(depth2d_ms<float, access::read> source [[texture(0)]],
        device uint *words [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]],
        uint3 position [[threadgroup_position_in_grid]]) {
    words[words_base(position, shape, 1u)] = as_type<uint>(source.read(position.xy, position.z));
}

// Metal leaves a stencil texture's other components undefined: only the first is read.
kernel void rt6_read_stencil(texture2d_ms<uint, access::read> source [[texture(0)]],
        device uint *words [[buffer(0)]], constant Rt6Shape &shape [[buffer(1)]],
        uint3 position [[threadgroup_position_in_grid]]) {
    words[words_base(position, shape, 1u)] = source.read(position.xy, position.z).x;
}
