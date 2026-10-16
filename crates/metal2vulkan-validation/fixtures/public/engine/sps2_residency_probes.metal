// Owned fixture: the residency word (`sparse_color::resident()`, the front
// end's `(status & 1) ^ 1`) and the value of a texture read, sample, and gather at
// authored coordinates. Thread `i` runs probe `i` of buffer 0: its kind picks the access, its
// texel `(x, y)` addresses a read, and its `uv` a sample or gather (normalized, level 0). Texture
// 0 is sampled (a read of it is a fetch), texture 1 has read access (a storage read). A case binds
// ordinary or sparse textures (`textures[].sparse`), so one module measures a texture that is not
// sparse, a sparse texture's mapped and unmapped tiles, a filter footprint across a tile edge, and
// accesses outside either. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_zero(filter::nearest, address::clamp_to_zero, coord::normalized);
constant sampler linear_zero(filter::linear, address::clamp_to_zero, coord::normalized);

struct Probe {
    uint kind;
    uint x;
    uint y;
    uint reserved;
    float2 uv;
    float2 reserved2;
};

// Probe `i`'s two words: the texel's raw bits, then (resident, kind, i, 0x5A5).
static void put(device uint4 *out, uint i, uint kind, sparse_color<float4> c)
{
    out[2 * i] = as_type<uint4>(c.value());
    out[2 * i + 1] = uint4(c.resident() ? 1u : 0u, kind, i, 0x5A5u);
}

kernel void sps2_residency_probes(texture2d<float, access::sample> t [[texture(0)]],
                                  texture2d<float, access::read> r [[texture(1)]],
                                  const device Probe *probes [[buffer(0)]],
                                  device uint4 *out [[buffer(1)]],
                                  uint i [[thread_position_in_grid]])
{
    Probe p = probes[i];
    uint2 texel = uint2(p.x, p.y);
    switch (p.kind) {
    case 0: put(out, i, 0, t.sparse_read(texel)); break;
    case 1: put(out, i, 1, r.sparse_read(texel)); break;
    case 2: put(out, i, 2, t.sparse_sample(nearest_zero, p.uv, level(0))); break;
    case 3: put(out, i, 3, t.sparse_sample(linear_zero, p.uv, level(0))); break;
    default: put(out, i, 4, t.sparse_gather(nearest_zero, p.uv)); break;
    }
}
