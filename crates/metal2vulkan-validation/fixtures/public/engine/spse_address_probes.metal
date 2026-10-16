// Owned fixture: the residency word (`sparse_color::resident()`) and value of
// a 2D texture's nearest and linear samples and gathers at level 0 through clamp_to_edge and
// clamp_to_border constexpr samplers, at authored coordinates (a probe buffer). Thread `i` runs
// probe `i` of buffer 0: its kind picks the sampler and access, and its `uv` the coordinate. A case
// binds a sparse texture whose texels at an edge are unmapped (`textures[].sparse`), so it asks
// whether a texel the addressing clamps to (the edge) or replaces (the border) counts toward
// residency. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_edge(filter::nearest, address::clamp_to_edge, coord::normalized);
constant sampler linear_edge(filter::linear, address::clamp_to_edge, coord::normalized);
constant sampler nearest_border(filter::nearest, address::clamp_to_border,
                                border_color::opaque_white, coord::normalized);
constant sampler linear_border(filter::linear, address::clamp_to_border,
                               border_color::opaque_white, coord::normalized);

struct Probe {
    uint kind;
    uint reserved;
    float2 uv;
    float4 reserved2;
};

// Probe `i`'s two words: the value's raw bits, then (resident, kind, i, 0x5A5).
static void put(device uint4 *out, uint i, uint kind, sparse_color<float4> c)
{
    out[2 * i] = as_type<uint4>(c.value());
    out[2 * i + 1] = uint4(c.resident() ? 1u : 0u, kind, i, 0x5A5u);
}

kernel void spse_address_probes(texture2d<float, access::sample> t [[texture(0)]],
                                const device Probe *probes [[buffer(0)]],
                                device uint4 *out [[buffer(1)]],
                                uint i [[thread_position_in_grid]])
{
    Probe p = probes[i];
    switch (p.kind) {
    case 0: put(out, i, 0, t.sparse_sample(nearest_edge, p.uv, level(0))); break;
    case 1: put(out, i, 1, t.sparse_sample(linear_edge, p.uv, level(0))); break;
    case 2: put(out, i, 2, t.sparse_gather(nearest_edge, p.uv)); break;
    case 3: put(out, i, 3, t.sparse_sample(nearest_border, p.uv, level(0))); break;
    case 4: put(out, i, 4, t.sparse_sample(linear_border, p.uv, level(0))); break;
    default: put(out, i, 5, t.sparse_gather(nearest_border, p.uv)); break;
    }
}
