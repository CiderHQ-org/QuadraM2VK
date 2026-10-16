// Owned fixture: what a write to a sparse texture's unmapped texel does. A
// thread writes a texel of a read_write 2D texture, fences the texture, and reads the texel back
// with its residency (`sparse_color::resident()`), so the thread observes its own write. Thread `i`
// runs probe `i` of buffer 0: kind 0 writes `value` at texel `(x, y)` and then reads it, kind 1
// only reads it. Each probe addresses its own texel, so no thread reads a texel another writes. A
// case binds a sparse texture with some tiles mapped (`textures[].sparse`): a write to a mapped
// texel, to an unmapped one, and past the texture, each read back. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Probe {
    uint kind;
    uint x;
    uint y;
    uint reserved;
    float4 value;
};

kernel void spse_write_probes(texture2d<float, access::read_write> w [[texture(0)]],
                              const device Probe *probes [[buffer(0)]],
                              device uint4 *out [[buffer(1)]],
                              uint i [[thread_position_in_grid]])
{
    Probe p = probes[i];
    uint2 texel = uint2(p.x, p.y);
    if (p.kind == 0) {
        w.write(p.value, texel);
        w.fence();
    }
    sparse_color<float4> c = w.sparse_read(texel);
    out[2 * i] = as_type<uint4>(c.value());
    out[2 * i + 1] = uint4(c.resident() ? 1u : 0u, p.kind, i, 0x5A5u);
}
