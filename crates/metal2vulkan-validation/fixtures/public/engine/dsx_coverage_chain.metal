// Owned fixture (written for this repository, no third-party source): a coverage chain
// through an RG16Float attachment, as a path renderer builds one. Instanced rectangles add their
// weights to the attachment's green under additive blending (`chain_edges`); a pass over each
// cell reads the sum back by framebuffer fetch under early fragment tests and leaves its absolute
// value, saturated, in red, green cleared (`chain_coverage`); a colour pass reads that coverage
// by framebuffer fetch, discards where it is 0 (`chain_color_discarding`, or without the discard
// `chain_color`), and writes its colour times the coverage to color(0), the coverage cleared.
// Every value is a dyadic fraction a `half` holds exactly. Loop-free.
//
// The sum may instead come from span quads (`chain_spans`, `chain_span_edges`), as a path
// renderer covers an outline: one quad per vertical edge of the outline, from the edge (less an
// outset) to a right bound (plus the outset), over the edge's rows (and the outset above and
// below), each fragment adding the edge's weight times the share of its pixel the edge's rows
// span and the share right of the edge.
//
// The chain may instead end in a gradient's fill (`chain_gradient`), as a path renderer's
// gradient ends one: it reads both the colour and the coverage by framebuffer fetch, discards
// where the coverage is 0, adds a 32x32 noise texture's texel at the pixel (sampled, a dither),
// discards there again, and writes the colour where the coverage is 1, else the colour mixed over
// the fetched one by the coverage. `chain_gradient_lod0` samples at LOD 0, `chain_gradient_hoisted`
// samples before the first discard.
#include <metal_stdlib>
using namespace metal;

struct ChainVertex {
    float4 position [[position]];
    float weight [[flat]];
};

// Instance `iid` covers the pixel rectangle `rects[iid]` (x0, y0, x1, y1, from the top left) of
// a target of `size` pixels, as a strip of four vertices, carrying `weights[iid]`.
vertex ChainVertex chain_rects(uint vid [[vertex_id]], uint iid [[instance_id]],
                               const device float4 *rects [[buffer(0)]],
                               const device float *weights [[buffer(1)]],
                               constant float2 &size [[buffer(2)]])
{
    float4 r = rects[iid];
    float2 corner = float2((vid & 1u) != 0u ? r.z : r.x, (vid & 2u) != 0u ? r.w : r.y);
    ChainVertex out;
    out.position = float4(corner.x / size.x * 2.0f - 1.0f, 1.0f - corner.y / size.y * 2.0f,
                          0.5f, 1.0f);
    out.weight = weights[iid];
    return out;
}

struct ChainCoverage {
    half2 coverage [[color(1)]];
};

struct ChainSpan {
    float4 position [[position]];
    float4 edge [[flat]];
};

// The span pass's frame: the target's size in pixels, the spans' right bound, how far a span
// reaches past its edge to the left, above and below, and how many edges there are.
struct ChainSpanFrame {
    float2 size;
    float right;
    float outset;
    uint count;
};

// Instance `iid` draws edges `13 iid` to `13 iid + 12`, four vertices an edge (`vid` through an
// index buffer): corner 0 at the edge's top, 1 at the bound's top, 2 at the bound's bottom, 3 at
// the edge's bottom. Edge `e` is `edges[e]` = (x, top row, bottom row, weight), in pixels from the
// top left. An edge at or past `count` collapses to (-2, -2).
vertex ChainSpan chain_spans(ushort vid [[vertex_id]], uint iid [[instance_id]],
                             const device float4 *edges [[buffer(0)]],
                             constant ChainSpanFrame &frame [[buffer(1)]])
{
    uint index = iid * 13u + uint(vid >> 2);
    ChainSpan out;
    if (index >= frame.count) {
        out.position = float4(-2.0f, -2.0f, 0.0f, 1.0f);
        out.edge = float4(0.0f);
        return out;
    }
    float4 e = edges[index];
    uint corner = uint(vid & 3);
    float x = (corner == 0u || corner == 3u) ? e.x - frame.outset : frame.right + frame.outset;
    float y = corner < 2u ? e.y - frame.outset : e.z + frame.outset;
    out.position = float4(x / frame.size.x * 2.0f - 1.0f, 1.0f - y / frame.size.y * 2.0f, 0.5f,
                          1.0f);
    out.edge = e;
    return out;
}

// Adds the edge's weight times the share of the pixel its rows span and the share right of it.
fragment ChainCoverage chain_span_edges(ChainSpan in [[stage_in]])
{
    float2 p = floor(in.position.xy);
    float rows = saturate(min(in.edge.z, p.y + 1.0f) - max(in.edge.y, p.y));
    float right = saturate(p.x + 1.0f - in.edge.x);
    ChainCoverage out;
    out.coverage = half2(0.0h, half(in.edge.w * rows * right));
    return out;
}

// Adds the instance's weight to green.
fragment ChainCoverage chain_edges(ChainVertex in [[stage_in]])
{
    ChainCoverage out;
    out.coverage = half2(0.0h, half(in.weight));
    return out;
}

// Leaves the sum's absolute value, saturated, in red, green cleared.
[[early_fragment_tests]] fragment ChainCoverage chain_coverage(half2 accumulated [[color(1)]])
{
    ChainCoverage out;
    out.coverage = half2(saturate(abs(accumulated.y)), 0.0h);
    return out;
}

struct ChainColor {
    half4 color [[color(0)]];
    half2 coverage [[color(1)]];
};

// Discards where the coverage is 0; writes the colour times the coverage, the coverage cleared.
fragment ChainColor chain_color_discarding(half2 coverage [[color(1)]],
                                           constant half4 &color [[buffer(0)]])
{
    if (coverage.x == 0.0h) {
        discard_fragment();
    }
    ChainColor out;
    out.color = color * coverage.x;
    out.coverage = half2(0.0h);
    return out;
}

// The same without the discard.
fragment ChainColor chain_color(half2 coverage [[color(1)]], constant half4 &color [[buffer(0)]])
{
    ChainColor out;
    out.color = color * coverage.x;
    out.coverage = half2(0.0h);
    return out;
}

constexpr sampler chain_noise_sampler(coord::normalized, address::repeat, filter::nearest);

// The noise texel at the pixel, the texture repeating over the target every 32 pixels.
static half3 chain_noise(texture2d<half> noise, float4 position)
{
    return noise.sample(chain_noise_sampler, position.xy * 0.03125f).rgb;
}

// The same at LOD 0.
static half3 chain_noise_lod0(texture2d<half> noise, float4 position)
{
    return noise.sample(chain_noise_sampler, position.xy * 0.03125f, level(0.0f)).rgb;
}

// The colour where the coverage `c` is 1, else the colour mixed over `dest` by it; the coverage
// times the colour's alpha.
static ChainColor chain_gradient_fill(half4 colour, half4 dest, half c)
{
    ChainColor out;
    out.color = c == 1.0h ? colour : mix(dest, colour, c);
    out.coverage = half2(c * colour.a, 0.0h);
    return out;
}

// Discards where the coverage is 0, adds the noise to the colour, discards there again, and fills.
fragment ChainColor chain_gradient(ChainVertex in [[stage_in]], half4 dest [[color(0)]],
                                   half2 coverage [[color(1)]],
                                   constant half4 &color [[buffer(0)]],
                                   texture2d<half> noise [[texture(0)]])
{
    half c = coverage.x;
    if (c == 0.0h) {
        discard_fragment();
    }
    half4 colour = color;
    colour.rgb += chain_noise(noise, in.position) * colour.a;
    if (c == 0.0h) {
        discard_fragment();
    }
    return chain_gradient_fill(colour, dest, c);
}

// The same, sampling at LOD 0.
fragment ChainColor chain_gradient_lod0(ChainVertex in [[stage_in]], half4 dest [[color(0)]],
                                        half2 coverage [[color(1)]],
                                        constant half4 &color [[buffer(0)]],
                                        texture2d<half> noise [[texture(0)]])
{
    half c = coverage.x;
    if (c == 0.0h) {
        discard_fragment();
    }
    half4 colour = color;
    colour.rgb += chain_noise_lod0(noise, in.position) * colour.a;
    if (c == 0.0h) {
        discard_fragment();
    }
    return chain_gradient_fill(colour, dest, c);
}

// The same, sampling before the first discard.
fragment ChainColor chain_gradient_hoisted(ChainVertex in [[stage_in]], half4 dest [[color(0)]],
                                           half2 coverage [[color(1)]],
                                           constant half4 &color [[buffer(0)]],
                                           texture2d<half> noise [[texture(0)]])
{
    half3 n = chain_noise(noise, in.position);
    half c = coverage.x;
    if (c == 0.0h) {
        discard_fragment();
    }
    half4 colour = color;
    colour.rgb += n * colour.a;
    if (c == 0.0h) {
        discard_fragment();
    }
    return chain_gradient_fill(colour, dest, c);
}
