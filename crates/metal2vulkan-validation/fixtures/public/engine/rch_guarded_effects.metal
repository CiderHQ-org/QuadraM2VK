// Owned fixture: fragment functions that discard and then make side
// effects, which a predicated discard runs only where the invocation was not discarded: a store
// (`rch_guarded_store`), an atomic whose old value picks the slot of a later store
// (`rch_guarded_atomic`), a store in a loop (`rch_guarded_loop`), a texture write
// (`rch_guarded_write`), a copy of a pair of vectors (`rch_guarded_copy`, through `copy_pair`,
// whose deferred body has Apple's compiler declare `llvm.memcpy` after the sample and the discard,
// where the opaque text, whose `llvm-dis` renames it, declares it too: the two texts then declare
// what the function calls in one order), and a store of a simdgroup sum
// (`rch_guarded_simd`), whose non-quad subgroup operation stops the predicate where
// the profile demotes. Each samples an 8x8 texture repeated over the target at LOD
// 0, discards where the texel's alpha is below 0.5, and returns the texel: the shape whose
// discard waits past the stores on a driver that drops a simdgroup's stores.
// `rch_guarded_vertex` draws the triangle whose three corners `buffer(0)` holds, in clip space.
// A pixel's index is `y * 64 + x`.
#include <metal_stdlib>
using namespace metal;

constexpr sampler guarded_sampler(filter::nearest, address::repeat);

struct GuardedIn {
    float4 position [[position]];
    float4 color;
};

struct GuardedPair {
    float4 a;
    float4 b;
};

vertex GuardedIn rch_guarded_vertex(uint vid [[vertex_id]],
                                    constant float2 *corners [[buffer(0)]])
{
    GuardedIn out;
    out.position = float4(corners[vid], 0.0f, 1.0f);
    out.color = float4(1.0f);
    return out;
}

static float4 shade(GuardedIn in, texture2d<float> tex)
{
    return in.color * tex.sample(guarded_sampler, in.position.xy * 0.125f, level(0.0f));
}

static uint pixel(GuardedIn in)
{
    uint2 p = uint2(in.position.xy);
    return p.y * 64u + p.x;
}

static void copy_pair(device const GuardedPair *from, device GuardedPair *to)
{
    *to = *from;
}

fragment float4 rch_guarded_store(GuardedIn in [[stage_in]],
                                  texture2d<float> tex [[texture(0)]],
                                  device float4 *out [[buffer(0)]])
{
    float4 c = shade(in, tex);
    if (c.a < 0.5f) {
        discard_fragment();
    }
    out[pixel(in)] = c * 0.5f;
    return c;
}

fragment float4 rch_guarded_atomic(GuardedIn in [[stage_in]],
                                   texture2d<float> tex [[texture(0)]],
                                   device atomic_uint *counter [[buffer(0)]],
                                   device uint *out [[buffer(1)]])
{
    float4 c = shade(in, tex);
    if (c.a < 0.5f) {
        discard_fragment();
    }
    uint slot = atomic_fetch_add_explicit(&counter[0], 1u, memory_order_relaxed);
    out[slot] = pixel(in);
    return c;
}

fragment float4 rch_guarded_loop(GuardedIn in [[stage_in]],
                                 texture2d<float> tex [[texture(0)]],
                                 device float *out [[buffer(0)]],
                                 constant uint &n [[buffer(1)]])
{
    float4 c = shade(in, tex);
    if (c.a < 0.5f) {
        discard_fragment();
    }
    for (uint i = 0u; i < n; i++) {
        out[i] = c.x * float(i);
    }
    return c;
}

fragment float4 rch_guarded_write(GuardedIn in [[stage_in]],
                                  texture2d<float> tex [[texture(0)]],
                                  texture2d<float, access::write> dst [[texture(1)]])
{
    float4 c = shade(in, tex);
    if (c.a < 0.5f) {
        discard_fragment();
    }
    dst.write(c * 0.25f, uint2(in.position.xy));
    return c;
}

fragment float4 rch_guarded_copy(GuardedIn in [[stage_in]],
                                 texture2d<float> tex [[texture(0)]],
                                 device GuardedPair *out [[buffer(0)]],
                                 device const GuardedPair *pairs [[buffer(1)]])
{
    float4 c = shade(in, tex);
    if (c.a < 0.5f) {
        discard_fragment();
    }
    copy_pair(&pairs[uint(c.x * 4.0f)], &out[pixel(in)]);
    return c;
}

fragment float4 rch_guarded_simd(GuardedIn in [[stage_in]],
                                 texture2d<float> tex [[texture(0)]],
                                 device float *out [[buffer(0)]])
{
    float4 c = shade(in, tex);
    if (c.a < 0.5f) {
        discard_fragment();
    }
    float total = simd_sum(c.x);
    out[uint(in.position.x) & 63u] = total;
    return c;
}
