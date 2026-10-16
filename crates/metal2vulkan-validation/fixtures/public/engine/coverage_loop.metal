// Owned fixture: fragment functions that read [[sample_mask]] and that Metal
// runs once per covered sample (each writes memory, and most read [[sample_id]]), drawn by the
// fullscreen triangle in a pass without attachments. Metal hands every run the coverage of the
// piece of the primitive the pixel's fragment came from; the compile layer runs such a function
// once per pixel and its body once per covered sample. The record and union functions fit that
// loop; the others each hold one thing it refuses. A pixel is word-indexed as in a 2x2 area, so
// a 1x1 pass (the oracle's without attachments) uses pixel 0 alone. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct CovPoint {
    float4 position [[position]];
};

struct CovVaryings {
    float4 position [[position]];
    float2 uv [[sample_perspective]];
};

// The fullscreen triangle, (-1, -1), (3, -1), (-1, 3), computed rather than indexed from a local
// array (whose lifetime markers LLVM majors print differently).
vertex CovPoint cov_cover(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return CovPoint { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

// The same triangle with uv = 2a, the generated companion's value for a float2 varying.
vertex CovVaryings cov_cover_uv(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return CovVaryings { float4(4.0 * a - 1.0, 0.0, 1.0), 2.0 * a };
}

// Word (pixel * 4 + sample) of a 2x2 area's pixels.
static uint cov_slot(float4 position, uint sample) {
    return (uint(position.y) * 2u + uint(position.x)) * 4u + sample;
}

// Each run ORs its coverage, marked by 0x100, into its sample's word, and counts itself 16 words
// later: every covered sample's word holds the coverage its run read, and its count is 1.
fragment void cov_loop_record(CovPoint in [[stage_in]], uint mask [[sample_mask]],
                              uint sample [[sample_id]], device atomic_uint *words [[buffer(0)]]) {
    uint slot = cov_slot(in.position, sample);
    atomic_fetch_or_explicit(&words[slot], mask | 0x100u, memory_order_relaxed);
    atomic_fetch_add_explicit(&words[16u + slot], 1u, memory_order_relaxed);
}

// What any split of a pixel's coverage leaves alone, in the pixel's four words: the union of the
// coverages its runs read, the union of their samples, the number of runs, and whether a run's
// sample was ever outside the coverage it read (0: never).
fragment void cov_loop_union(CovPoint in [[stage_in]], uint mask [[sample_mask]],
                             uint sample [[sample_id]], device atomic_uint *words [[buffer(0)]]) {
    uint base = cov_slot(in.position, 0u);
    atomic_fetch_or_explicit(&words[base], mask, memory_order_relaxed);
    atomic_fetch_or_explicit(&words[base + 1u], 1u << sample, memory_order_relaxed);
    atomic_fetch_add_explicit(&words[base + 2u], 1u, memory_order_relaxed);
    atomic_fetch_or_explicit(&words[base + 3u], ((mask >> sample) & 1u) ^ 1u,
                             memory_order_relaxed);
}

// A sample-qualified varying read at each run's sample, beside the coverage: the run's sample's
// word ORs uv * 16 rounded (x in bits 0-7, y in bits 8-15), the coverage in bits 16-23, and bit 24.
fragment void cov_loop_varying(CovVaryings in [[stage_in]], uint mask [[sample_mask]],
                               uint sample [[sample_id]], device atomic_uint *words [[buffer(0)]]) {
    uint2 at = uint2(in.uv * 16.0 + 0.5);
    uint word = (at.x & 0xffu) | ((at.y & 0xffu) << 8) | ((mask & 0xffu) << 16) | (1u << 24);
    atomic_fetch_or_explicit(&words[cov_slot(in.position, sample)], word, memory_order_relaxed);
}

// Refused by the loop: early fragment tests, which run only the samples that pass while each run
// reads the coverage before the test.
[[early_fragment_tests]]
fragment void cov_loop_early(CovPoint in [[stage_in]], uint mask [[sample_mask]],
                             uint sample [[sample_id]], device atomic_uint *words [[buffer(0)]]) {
    atomic_fetch_or_explicit(&words[cov_slot(in.position, sample)], mask | 0x100u,
                             memory_order_relaxed);
}

// Refused by the loop: a discard, which on Metal ends one sample's run.
fragment void cov_loop_discard(CovPoint in [[stage_in]], uint mask [[sample_mask]],
                               uint sample [[sample_id]], device atomic_uint *words [[buffer(0)]]) {
    if (((mask >> sample) & 1u) == 0u) {
        discard_fragment();
    }
    atomic_fetch_or_explicit(&words[cov_slot(in.position, sample)], mask | 0x100u,
                             memory_order_relaxed);
}

// Refused by the loop: a derivative, whose quad is the sample's on Metal.
fragment void cov_loop_derivative(CovPoint in [[stage_in]], uint mask [[sample_mask]],
                                  uint sample [[sample_id]],
                                  device atomic_uint *words [[buffer(0)]]) {
    uint slope = as_type<uint>(dfdx(in.position.x));
    atomic_fetch_or_explicit(&words[cov_slot(in.position, sample)], mask ^ slope,
                             memory_order_relaxed);
}

// Refused by the loop: a simdgroup operation, whose simdgroup holds the sample's runs on Metal.
fragment void cov_loop_simdgroup(CovPoint in [[stage_in]], uint mask [[sample_mask]],
                                 uint sample [[sample_id]],
                                 device atomic_uint *words [[buffer(0)]]) {
    uint lanes = simd_sum(1u);
    atomic_fetch_or_explicit(&words[cov_slot(in.position, sample)], mask | (lanes << 8),
                             memory_order_relaxed);
}

// Refused by the loop: a color attachment read, which on Metal is the sample's.
fragment void cov_loop_fetch(CovPoint in [[stage_in]], uint mask [[sample_mask]],
                             uint sample [[sample_id]], uint prior [[color(0)]],
                             device atomic_uint *words [[buffer(0)]]) {
    atomic_fetch_or_explicit(&words[cov_slot(in.position, sample)], mask | (prior << 8),
                             memory_order_relaxed);
}
