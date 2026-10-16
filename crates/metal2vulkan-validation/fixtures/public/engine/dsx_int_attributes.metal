// Owned fixture: unnormalized 8- and 16-bit integer vertex formats read
// into float attributes, which Metal converts to each component's integer value.
// - `int_attribute_to_float4`, `int_attribute_to_half4`: a rasterization-disabled point per
//   vertex; vertex v writes the float4 attribute (the half4 one widened, exactly) it fetched to
//   out[v].
// - `label_vertex`, `label_fragment`: a text label pipeline's shape. Eight attributes (a float2
//   corner, a ushort2 read as float2 normalized and not, two uchar4 colors read as half4, a
//   normalized ushort2 scale, a float2 texture coordinate and a float2 shift); nine varyings
//   beside the position, of mixed interpolation (center perspective, center no-perspective,
//   centroid perspective); a fragment that samples a texture and checks each varying against
//   the values its buffer holds: red 1 where the integer counts hold, green 0 where the
//   normalized values and the colors hold, blue the texture's red, alpha 1. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Float4Attribute {
    float4 a [[attribute(0)]];
};

struct Half4Attribute {
    half4 a [[attribute(0)]];
};

vertex void int_attribute_to_float4(Float4Attribute in [[stage_in]],
                                    device float4 *out [[buffer(0)]],
                                    uint vid [[vertex_id]]) {
    out[vid] = in.a;
}

vertex void int_attribute_to_half4(Half4Attribute in [[stage_in]],
                                   device float4 *out [[buffer(0)]],
                                   uint vid [[vertex_id]]) {
    out[vid] = float4(in.a);
}

struct LabelIn {
    float2 corner [[attribute(0)]];
    float2 unit [[attribute(1)]];
    float2 count [[attribute(2)]];
    half4 tint [[attribute(3)]];
    half4 halo [[attribute(4)]];
    float2 scale [[attribute(5)]];
    float2 uv [[attribute(6)]];
    float2 shift [[attribute(7)]];
};

struct LabelOut {
    float4 position [[position]];
    float2 uv;
    half4 tint;
    half4 halo [[center_no_perspective]];
    float count_x;
    float count_y [[centroid_perspective]];
    float2 unit;
    float2 scale [[center_no_perspective]];
    half4 count_half;
    half4 shift_half;
};

struct LabelExpected {
    float2 count;
    float2 unit;
    float2 scale;
    float2 shift;
    float4 tint;
    float4 halo;
    float4 count_half;
};

vertex LabelOut label_vertex(LabelIn in [[stage_in]]) {
    LabelOut out;
    out.position = float4(in.corner + in.shift, 0.5, 1.0);
    out.uv = in.uv;
    out.tint = in.tint;
    out.halo = in.halo;
    out.count_x = in.count.x;
    out.count_y = in.count.y;
    out.unit = in.unit;
    out.scale = in.scale;
    out.count_half = half4(half2(in.count), half2(in.unit));
    out.shift_half = half4(half2(in.shift * 4.0), 0.0h, 1.0h);
    return out;
}

static bool near(float a, float b, float within) {
    return abs(a - b) < within;
}

fragment half4 label_fragment(LabelOut in [[stage_in]],
                              constant LabelExpected &e [[buffer(0)]],
                              texture2d<half> glyphs [[texture(0)]],
                              sampler s [[sampler(0)]]) {
    bool count = near(in.count_x, e.count.x, 0.25) && near(in.count_y, e.count.y, 0.25)
        && near(float(in.count_half.x), e.count_half.x, 0.25)
        && near(float(in.count_half.y), e.count_half.y, 0.25);
    bool unit = all(abs(in.unit - e.unit) < 1.0 / 1024.0)
        && all(abs(in.scale - e.scale) < 1.0 / 1024.0)
        && all(abs(float2(in.count_half.zw) - e.count_half.zw) < 1.0 / 512.0)
        && all(abs(float2(in.shift_half.xy) - 4.0 * e.shift) < 1.0 / 64.0);
    bool colors = all(abs(float4(in.tint) - e.tint) < 1.0 / 512.0)
        && all(abs(float4(in.halo) - e.halo) < 1.0 / 512.0);
    half glyph = glyphs.sample(s, in.uv).r;
    half inside = in.position.x >= 0.0 ? 1.0h : 0.0h;
    return half4(count ? 1.0h : 0.0h, unit && colors ? 0.0h : 1.0h, glyph, inside);
}
