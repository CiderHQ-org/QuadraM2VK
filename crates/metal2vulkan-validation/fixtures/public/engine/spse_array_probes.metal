// Owned fixture: the residency word (`sparse_color::resident()`) and value of
// a 2D array texture's fetch at a layer and level, and of its nearest and linear samples at a layer
// and level and gather at a layer, at authored coordinates. Thread `i` runs probe `i` of buffer 0:
// its kind picks the access, its texel `(x, y)`, `layer`, and level `lod` address a read, and its
// `uv`, `layer`, and float level `lodf` a sample (a gather reads level 0). Texture 0 is sampled, so
// a read of it is a fetch. A case binds a sparse 2D array texture (`textures[].sparse`) whose slices
// are mapped apart, so one module measures each slice's residency and a layer or level past the
// texture's count. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_zero(filter::nearest, address::clamp_to_zero, coord::normalized);
constant sampler linear_zero(filter::linear, address::clamp_to_zero, coord::normalized);

struct Probe {
    uint kind;
    uint x;
    uint y;
    uint layer;
    float2 uv;
    float lodf;
    uint lod;
};

// Probe `i`'s two words: the texel's raw bits, then (resident, kind, i, 0x5A5).
static void put(device uint4 *out, uint i, uint kind, sparse_color<float4> c)
{
    out[2 * i] = as_type<uint4>(c.value());
    out[2 * i + 1] = uint4(c.resident() ? 1u : 0u, kind, i, 0x5A5u);
}

kernel void spse_array_probes(texture2d_array<float, access::sample> a [[texture(0)]],
                              const device Probe *probes [[buffer(0)]],
                              device uint4 *out [[buffer(1)]],
                              uint i [[thread_position_in_grid]])
{
    Probe p = probes[i];
    switch (p.kind) {
    case 0: put(out, i, 0, a.sparse_read(uint2(p.x, p.y), p.layer, p.lod)); break;
    case 1: put(out, i, 1, a.sparse_sample(nearest_zero, p.uv, p.layer, level(p.lodf))); break;
    case 2: put(out, i, 2, a.sparse_sample(linear_zero, p.uv, p.layer, level(p.lodf))); break;
    default: put(out, i, 3, a.sparse_gather(nearest_zero, p.uv, p.layer)); break;
    }
}
