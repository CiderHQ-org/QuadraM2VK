// metal-std: metal4.0
// metal-macos-min: 27.0
// Owned fixture: the AIR 2.9 twin of
// validation/fixtures/public/engine/spse_write_barrier.metal. Below the next blank line is that
// file, unchanged; the directives above compile it as MSL 4.0 for macOS 27, where Apple's compiler
// writes AIR 2.9 (each memory atomic with its `mem_flags`, each barrier with its `memory_order`).
// It compiles on every committed profile to the same words as its original
// (tests/engine_air_2_9.rs), so the original's cases and rows speak for it.

// Owned fixture: a thread's own write to a read_write 2D texture, read back
// after a texture fence and a threadgroup barrier on texture memory, sparse or not. Thread `i` runs
// probe `i` of buffer 0: kind 0 writes `value` at texel `(x, y)` and fences the texture, kind 1
// does not; every thread then reaches `threadgroup_barrier(mem_flags::mem_texture)` (outside the
// branch) and reads the texel back with its residency (`sparse_color::resident()`). Each probe
// addresses its own texel, so no thread reads a texel another writes. A case binds the texture
// sparse with some tiles mapped (`textures[].sparse`), or not sparse as the control of the same
// probes. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Probe {
    uint kind;
    uint x;
    uint y;
    uint reserved;
    float4 value;
};

kernel void spse_write_barrier(texture2d<float, access::read_write> w [[texture(0)]],
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
    threadgroup_barrier(mem_flags::mem_texture);
    sparse_color<float4> c = w.sparse_read(texel);
    out[2 * i] = as_type<uint4>(c.value());
    out[2 * i + 1] = uint4(c.resident() ? 1u : 0u, p.kind, i, 0x5A5u);
}
