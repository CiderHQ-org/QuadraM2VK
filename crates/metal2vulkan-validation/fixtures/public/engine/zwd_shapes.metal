// Owned fixture: a compositor's antialiased shapes, as a window server draws an alert
// panel's rounded corners and a title bar's round buttons by their coverage. `zwd_quad` places two
// triangles over a shape's box given in pixels, two pixels wider each way for its soft edge, in a
// target of the given size. Each fragment function returns the shape's premultiplied color times
// its coverage at the pixel center, from a signed distance: `zwd_panel` a rounded rectangle's,
// `zwd_disc` a filled circle's, `zwd_ring` a circle's outline. Function constant 0 (`zwd_variant`)
// adds to the radius (`variant % 7`) and widens the soft edge (`1 + (variant / 7) / 2` pixels), so
// each value is a module of its own. `zwd_masked` returns the fill times a mask texture's sample
// at the box's coordinates, its channels weighted by the outline color (which picks the mask's
// channel); `zwd_mask_half` takes its `half` sample, `zwd_mask_gather` a `float` gather of its first
// channel, `zwd_mask_read` its `half` read, `zwd_mask_nearest` its `half` sample through a constexpr
// nearest sampler. `zwd_clip` discards outside a rounded rectangle (a stencil clip), `zwd_two` fills
// two color attachments, and `zwd_layer` samples twelve textures, two of them 3D tables. The quad lies at the depth `shape.w`. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

constant uint zwd_variant [[function_constant(0)]];

struct ZwdShape {
    float4 rect;  // x0, y0, x1, y1 in pixels
    float4 color; // premultiplied fill
    float4 ring;  // premultiplied outline
    float4 shape; // radius, target width, target height, depth
};

struct ZwdOut {
    float4 position [[position]];
};

vertex ZwdOut zwd_quad(uint vid [[vertex_id]], constant ZwdShape &s [[buffer(0)]])
{
    const float2 corners[6] = {
        float2(0.0f, 0.0f), float2(1.0f, 0.0f), float2(0.0f, 1.0f),
        float2(1.0f, 0.0f), float2(1.0f, 1.0f), float2(0.0f, 1.0f),
    };
    float2 t = corners[vid];
    float2 p = select(s.rect.xy - 2.0f, s.rect.zw + 2.0f, t > 0.5f);
    float2 size = s.shape.yz;
    ZwdOut out;
    out.position = float4(p.x / size.x * 2.0f - 1.0f, 1.0f - p.y / size.y * 2.0f, s.shape.w, 1.0f);
    return out;
}

static float zwd_radius(constant ZwdShape &s)
{
    return s.shape.x + float(zwd_variant % 7u);
}

static float zwd_soft()
{
    return 1.0f + float(zwd_variant / 7u) * 0.5f;
}

static float zwd_coverage(float d)
{
    return clamp(0.5f - d / zwd_soft(), 0.0f, 1.0f);
}

fragment float4 zwd_panel(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]])
{
    float r = zwd_radius(s);
    float2 c = (s.rect.xy + s.rect.zw) * 0.5f;
    float2 h = (s.rect.zw - s.rect.xy) * 0.5f;
    float2 q = abs(in.position.xy - c) - h + r;
    float d = length(max(q, 0.0f)) + min(max(q.x, q.y), 0.0f) - r;
    return s.color * zwd_coverage(d);
}

fragment float4 zwd_disc(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]])
{
    float2 c = (s.rect.xy + s.rect.zw) * 0.5f;
    float d = length(in.position.xy - c) - zwd_radius(s);
    return s.color * zwd_coverage(d);
}

fragment float4 zwd_ring(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]])
{
    float2 c = (s.rect.xy + s.rect.zw) * 0.5f;
    float d = abs(length(in.position.xy - c) - zwd_radius(s)) - 0.75f;
    return s.ring * zwd_coverage(d);
}

fragment float4 zwd_masked(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]],
                           texture2d<float> mask [[texture(0)]], sampler smp [[sampler(0)]])
{
    // The variant's term is far below a coordinate's step: a module per value, the same samples.
    float2 uv = (in.position.xy - s.rect.xy) / (s.rect.zw - s.rect.xy) + float(zwd_variant) * 1e-30f;
    return s.color * dot(mask.sample(smp, uv), s.ring);
}

// The mask's `half` sample (a compositor's mask texture, its format's class read as `half`), and
// a `float` gather of its first channel, each weighted by the outline color, times the fill.
fragment float4 zwd_mask_half(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]],
                              texture2d<half> mask [[texture(0)]], sampler smp [[sampler(0)]])
{
    float2 uv = (in.position.xy - s.rect.xy) / (s.rect.zw - s.rect.xy) + float(zwd_variant) * 1e-30f;
    half4 m = mask.sample(smp, uv);
    return s.color * float(dot(m, half4(s.ring)));
}

fragment float4 zwd_mask_gather(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]],
                                texture2d<float> mask [[texture(0)]], sampler smp [[sampler(0)]])
{
    float2 uv = (in.position.xy - s.rect.xy) / (s.rect.zw - s.rect.xy) + float(zwd_variant) * 1e-30f;
    return s.color * dot(mask.gather(smp, uv), s.ring);
}

// The mask's `half` read at the pixel (its 16 x 16 texels tiled), and its `half` sample through a
// constexpr nearest sampler, each weighted by the outline color, times the fill.
fragment float4 zwd_mask_read(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]],
                              texture2d<half> mask [[texture(0)]])
{
    uint2 p = uint2(in.position.xy - s.rect.xy) % 16u + uint2(zwd_variant / 1000u);
    return s.color * float(dot(mask.read(p), half4(s.ring)));
}

fragment float4 zwd_mask_nearest(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]],
                                 texture2d<half> mask [[texture(0)]])
{
    constexpr sampler near(filter::nearest, address::clamp_to_edge);
    float2 uv = (in.position.xy - s.rect.xy) / (s.rect.zw - s.rect.xy) + float(zwd_variant) * 1e-30f;
    return s.color * float(dot(mask.sample(near, uv), half4(s.ring)));
}

// A stencil clip's shape: the box's rounded rectangle, its corners circles of the radius
// `shape.x` tested at the pixel center (half-integers squared, exact); outside it, discarded.
fragment void zwd_clip(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]])
{
    float r = s.shape.x;
    float2 p = in.position.xy;
    float2 q = p - clamp(p, s.rect.xy + r, s.rect.zw - r);
    if (q.x * q.x + q.y * q.y > r * r) {
        discard_fragment();
    }
}

struct ZwdTwo {
    float4 first [[color(0)]];
    float2 second [[color(1)]];
};

// A fill of two color attachments: the fill to the first, the outline's first two channels to
// the second.
fragment ZwdTwo zwd_two(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]])
{
    ZwdTwo out;
    out.first = s.color;
    out.second = s.ring.xy;
    return out;
}

// A compositor's offscreen layer: the box's coordinates sample ten textures (five read as `half`,
// five as `float`) through one linear sampler, and their mix looks itself up in two 3D tables;
// the tables' mean times the fill, at half precision.
fragment half4 zwd_layer(ZwdOut in [[stage_in]], constant ZwdShape &s [[buffer(0)]],
                         texture2d<half> t0 [[texture(0)]], texture2d<half> t1 [[texture(1)]],
                         texture2d<half> t2 [[texture(2)]], texture2d<half> t3 [[texture(3)]],
                         texture2d<half> t4 [[texture(4)]], texture2d<float> t5 [[texture(5)]],
                         texture2d<float> t6 [[texture(6)]], texture2d<float> t7 [[texture(7)]],
                         texture2d<float> t8 [[texture(8)]], texture2d<float> t9 [[texture(9)]],
                         texture3d<half> lut0 [[texture(10)]], texture3d<half> lut1 [[texture(11)]],
                         sampler smp [[sampler(0)]])
{
    float2 uv = (in.position.xy - s.rect.xy) / (s.rect.zw - s.rect.xy) + float(zwd_variant) * 1e-30f;
    half4 h = (t0.sample(smp, uv) + t1.sample(smp, uv.yx) + t2.sample(smp, 1.0f - uv) +
               t3.sample(smp, uv * 0.5f) + t4.sample(smp, uv * 0.5f + 0.5f)) * 0.2h;
    float4 f = (t5.sample(smp, uv) + t6.sample(smp, uv.yx) + t7.sample(smp, 1.0f - uv) +
                t8.sample(smp, uv * 0.5f) + t9.sample(smp, uv * 0.5f + 0.5f)) * 0.2f;
    float3 c = mix(float3(h.rgb), f.rgb, 0.5f);
    half4 g0 = lut0.sample(smp, c);
    half4 g1 = lut1.sample(smp, float3(c.zy, float(h.a)));
    return half4(s.color) * ((g0 + g1) * 0.5h);
}

// A vertex function of a state without rasterization: it returns nothing and writes nothing.
vertex void zwd_none(uint vid [[vertex_id]], constant ZwdShape &s [[buffer(0)]])
{
}

// A path's exterior, as a compositor covers the side of a path's edge: each vertex hands the
// fragment the edge (`path_slope`, `path_intercept`), its own row (`path_y`) and a value; the
// fragment covers the pixel by its center's distance past the edge at its row, at half precision,
// into two color attachments. Four vertex functions hand them over (`zwd_path_v0` to
// `zwd_path_v3`: other clip-space `w`, other values, one with an extra varying first and its
// outputs in another order); three fragment functions read them (`zwd_path_exterior`
// perspective-correct, `zwd_path_exterior_flat` the edge flat, `zwd_path_exterior_linear` it
// without perspective).
struct ZwdPathV {
    float4 position [[position]];
    float path_y [[user(path_y)]];
    float path_slope [[user(path_slope)]];
    float path_intercept [[user(path_intercept)]];
    float path_value [[user(path_value)]];
};

struct ZwdPathV2 {
    float4 position [[position]];
    float4 extra [[user(path_extra)]];
    float path_value [[user(path_value)]];
    float path_intercept [[user(path_intercept)]];
    float path_slope [[user(path_slope)]];
    float path_y [[user(path_y)]];
};

static float4 zwd_path_position(uint vid, constant ZwdShape &s, float w)
{
    const float2 corners[6] = {
        float2(0.0f, 0.0f), float2(1.0f, 0.0f), float2(0.0f, 1.0f),
        float2(1.0f, 0.0f), float2(1.0f, 1.0f), float2(0.0f, 1.0f),
    };
    float2 t = corners[vid];
    float2 p = select(s.rect.xy - 2.0f, s.rect.zw + 2.0f, t > 0.5f);
    float2 size = s.shape.yz;
    return float4(p.x / size.x * 2.0f - 1.0f, 1.0f - p.y / size.y * 2.0f, s.shape.w, 1.0f) * w;
}

static float2 zwd_path_corner(uint vid)
{
    const float2 corners[6] = {
        float2(0.0f, 0.0f), float2(1.0f, 0.0f), float2(0.0f, 1.0f),
        float2(1.0f, 0.0f), float2(1.0f, 1.0f), float2(0.0f, 1.0f),
    };
    return corners[vid];
}

vertex ZwdPathV zwd_path_v0(uint vid [[vertex_id]], constant ZwdShape &s [[buffer(0)]])
{
    float2 t = zwd_path_corner(vid);
    ZwdPathV out;
    out.position = zwd_path_position(vid, s, 1.0f);
    out.path_y = mix(s.rect.y, s.rect.w, t.y);
    out.path_slope = s.ring.x + s.ring.y * t.x;
    out.path_intercept = mix(s.rect.x, s.rect.z, s.ring.z) + s.ring.w * t.y;
    out.path_value = 1.0f - 0.25f * t.x * t.y;
    return out;
}

vertex ZwdPathV zwd_path_v1(uint vid [[vertex_id]], constant ZwdShape &s [[buffer(0)]])
{
    float2 t = zwd_path_corner(vid);
    ZwdPathV out;
    out.position = zwd_path_position(vid, s, 1.0f + t.y);
    out.path_y = mix(s.rect.y, s.rect.w, t.y);
    out.path_slope = s.ring.x + s.ring.y * t.x;
    out.path_intercept = mix(s.rect.x, s.rect.z, s.ring.z) + s.ring.w * t.y;
    out.path_value = 1.0f - 0.25f * t.x * t.y;
    return out;
}

vertex ZwdPathV2 zwd_path_v2(uint vid [[vertex_id]], constant ZwdShape &s [[buffer(0)]])
{
    float2 t = zwd_path_corner(vid);
    ZwdPathV2 out;
    out.position = zwd_path_position(vid, s, 2.0f - t.x);
    out.extra = float4(t, 1.0f - t);
    out.path_value = 0.75f + 0.25f * t.y;
    out.path_intercept = mix(s.rect.x, s.rect.z, s.ring.z) - s.ring.w * t.x;
    out.path_slope = s.ring.x - s.ring.y * t.y;
    out.path_y = mix(s.rect.y, s.rect.w, t.y);
    return out;
}

vertex ZwdPathV zwd_path_v3(uint vid [[vertex_id]], constant ZwdShape &s [[buffer(0)]])
{
    float2 t = zwd_path_corner(vid);
    ZwdPathV out;
    out.position = zwd_path_position(vid, s, 1.0f);
    out.path_y = mix(s.rect.y, s.rect.w, 1.0f - t.y);
    out.path_slope = 0.5f * s.ring.x;
    out.path_intercept = mix(s.rect.x, s.rect.z, 1.0f - s.ring.z);
    out.path_value = 0.5f + 0.5f * t.x;
    return out;
}

struct ZwdPathIn {
    float4 position [[position]];
    float path_y [[user(path_y)]];
    float path_slope [[user(path_slope)]];
    float path_intercept [[user(path_intercept)]];
    float path_value [[user(path_value)]];
};

struct ZwdPathInFlat {
    float4 position [[position]];
    float path_y [[user(path_y)]];
    float path_slope [[user(path_slope), flat]];
    float path_intercept [[user(path_intercept), flat]];
    float path_value [[user(path_value)]];
};

struct ZwdPathInLinear {
    float4 position [[position]];
    float path_y [[user(path_y), center_no_perspective]];
    float path_slope [[user(path_slope), center_no_perspective]];
    float path_intercept [[user(path_intercept), center_no_perspective]];
    float path_value [[user(path_value), center_no_perspective]];
};

struct ZwdPathOut {
    half4 first [[color(0)]];
    half2 second [[color(1)]];
};

static ZwdPathOut zwd_path(float4 frag, float y, float slope, float intercept, float value,
                           constant ZwdShape &s)
{
    float edge = intercept + slope * (frag.y - y) + slope * (y - s.rect.y);
    half coverage = half(clamp(frag.x - edge + 0.5f, 0.0f, 1.0f)) * half(value);
    ZwdPathOut out;
    out.first = half4(s.color) * coverage;
    out.second = half2(s.ring.xy) * coverage;
    return out;
}

fragment ZwdPathOut zwd_path_exterior(ZwdPathIn in [[stage_in]], constant ZwdShape &s [[buffer(0)]])
{
    return zwd_path(in.position, in.path_y, in.path_slope, in.path_intercept, in.path_value, s);
}

fragment ZwdPathOut zwd_path_exterior_flat(ZwdPathInFlat in [[stage_in]],
                                           constant ZwdShape &s [[buffer(0)]])
{
    return zwd_path(in.position, in.path_y, in.path_slope, in.path_intercept, in.path_value, s);
}

fragment ZwdPathOut zwd_path_exterior_linear(ZwdPathInLinear in [[stage_in]],
                                             constant ZwdShape &s [[buffer(0)]])
{
    return zwd_path(in.position, in.path_y, in.path_slope, in.path_intercept, in.path_value, s);
}
