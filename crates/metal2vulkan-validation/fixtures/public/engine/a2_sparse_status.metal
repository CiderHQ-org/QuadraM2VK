// Owned fixture: the `{T, i8}` status word of a texture sample, gather, and
// read. MSL exposes it only through `sparse_color<T>::resident()`, which the front end
// spells `(status & 1) ^ 1`: bit 0 of the word is set when the texel is not resident. The case
// model binds ordinary (non-sparse) textures only, so the reading is what Metal reports for them,
// beside each operation's value. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nearest_clamp(filter::nearest, address::clamp_to_edge, coord::normalized);

kernel void a2_sparse_status(texture2d<float, access::sample> s [[texture(0)]],
                             texture2d<float, access::read> r [[texture(1)]],
                             device uint4 *out [[buffer(0)]],
                             uint tid [[thread_position_in_grid]])
{
    uint2 texel = uint2(tid & 3, tid >> 2);
    float2 uv = (float2(texel) + 0.5f) / 4.0f;
    sparse_color<float4> a = s.sparse_sample(nearest_clamp, uv, level(0));
    sparse_color<float4> g = s.sparse_gather(nearest_clamp, uv);
    sparse_color<float4> b = r.sparse_read(texel);
    out[4 * tid + 0] = as_type<uint4>(a.value());
    out[4 * tid + 1] = as_type<uint4>(g.value());
    out[4 * tid + 2] = as_type<uint4>(b.value());
    out[4 * tid + 3] = uint4(a.resident() ? 1u : 0u, g.resident() ? 1u : 0u,
                             b.resident() ? 1u : 0u, 0xA2u);
}
