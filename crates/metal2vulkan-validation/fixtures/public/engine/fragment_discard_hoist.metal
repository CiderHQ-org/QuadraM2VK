// metal-std: macos-metal2.3
// Owned fixture: a simdgroup count before a discard of
// whole quads, in the thirteen shapes measured for where Metal's compiler places it, past the
// discard (`hoisted_*`) or ahead of it, counting the dropped quads (`counted_*`), as Apple M4 and
// M3 Max draw them. Written for this repository; no third-party source. One loop (`hoisted_loop`,
// bounded by a buffer value).
#include <metal_stdlib>
using namespace metal;

struct DiscardHoistVertex {
    float4 position [[position]];
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3).
vertex DiscardHoistVertex discard_hoist_full(uint vid [[vertex_id]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    DiscardHoistVertex out;
    out.position = float4(4.0 * a - 1.0, 0.5, 1.0);
    return out;
}

__attribute__((noinline)) int discard_hoist_helper(int x) {
    return x * 7;
}

// Over an 8x4 target, bit `(y / 2) * 4 + x / 2` of `kept_quads` keeps the quad of pixel (x, y);
// every other quad is discarded whole. `k` is 0, so each condition below holds where its quad is
// not kept. `before` counts the simdgroup's lanes ahead of the discard; `after` and `kept` count
// them after it, `kept` those whose quad is kept.
#define BASE DiscardHoistVertex in [[stage_in]], constant uint &kept_quads [[buffer(0)]], \
    constant int &k [[buffer(1)]]

#define PROLOGUE \
    uint2 p = uint2(in.position.xy); \
    int one = k + 1; \
    bool quad_kept = ((kept_quads >> ((p.y >> 1) * 4u + (p.x >> 1))) & 1u) != 0u; \
    int before = simd_sum(1);

// A kept thread writes values that hold however a device packs quads into simdgroups: whether
// `before` counted lanes that left at the discard, `after - kept` (0), the count's remainder in
// whole quads (0), and whether `before` counted fewer.
#define COUNTS \
    int after = simd_sum(one); \
    int kept = simd_sum(quad_kept ? one : 0);

#define RESULT int4(before > after ? 1 : 0, after - kept, (before - after) & 3, before < after ? 1 : 0)

#define EPILOGUE \
    COUNTS \
    return RESULT;

fragment int4 hoisted_known(BASE) {
    PROLOGUE
    if (!quad_kept) {
        discard_fragment();
    }
    EPILOGUE
}

fragment int4 counted_dependent(BASE) {
    PROLOGUE
    if (!quad_kept && before > k) {
        discard_fragment();
    }
    EPILOGUE
}

fragment int4 hoisted_const(BASE) {
    PROLOGUE
    int sel = k;
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    EPILOGUE
}

fragment int4 hoisted_device_read(BASE, const device int *ro [[buffer(2)]]) {
    PROLOGUE
    int sel = ro[0];
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    EPILOGUE
}

fragment int4 hoisted_written_read(BASE, device int *rw [[buffer(3)]]) {
    PROLOGUE
    int sel = rw[0];
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    COUNTS
    rw[16u + p.y * 8u + p.x] = after;
    return RESULT;
}

fragment int4 hoisted_sample(BASE, texture2d<float> tex [[texture(0)]],
                             sampler smp [[sampler(0)]]) {
    PROLOGUE
    int sel = int(tex.sample(smp, float2(0.5)).x);
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    EPILOGUE
}

fragment int4 hoisted_texture_read(BASE, texture2d<float> tex [[texture(0)]]) {
    PROLOGUE
    int sel = int(tex.read(uint2(0, 0)).x);
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    EPILOGUE
}

fragment int4 hoisted_loop(BASE) {
    PROLOGUE
    int sel = 0;
    for (int i = 0; i < k + 4; i++) {
        sel += i * k;
    }
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    EPILOGUE
}

fragment int4 hoisted_call(BASE) {
    PROLOGUE
    int sel = discard_hoist_helper(k);
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    EPILOGUE
}

fragment int4 counted_call_use(BASE, device int *rw [[buffer(3)]]) {
    PROLOGUE
    int h = discard_hoist_helper(before);
    int sel = k;
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    COUNTS
    if (h == 12345) {
        rw[1] = h;
    }
    return RESULT;
}

// Metal moves the sum past this nested discard, but counted the dropped quads with the same
// nesting around another inner condition: its placement is unknown.
fragment int4 hoisted_nested(BASE) {
    PROLOGUE
    int sel = k;
    if (p.x < 100u + uint(k)) {
        if (!quad_kept && sel == 0) {
            discard_fragment();
        }
    }
    EPILOGUE
}

fragment int4 counted_store(BASE, device int *rw [[buffer(3)]]) {
    PROLOGUE
    rw[64u + p.y * 8u + p.x] = before;
    int sel = k;
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    EPILOGUE
}

fragment int4 counted_atomic(BASE, device atomic_int *atom [[buffer(4)]]) {
    PROLOGUE
    atomic_fetch_add_explicit(&atom[0], 1, memory_order_relaxed);
    int sel = k;
    if (!quad_kept && sel == 0) {
        discard_fragment();
    }
    EPILOGUE
}
