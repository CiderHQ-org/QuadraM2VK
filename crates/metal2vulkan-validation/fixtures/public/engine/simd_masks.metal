// Owned fixture: quad_ballot, simd_active_threads_mask, and a fragment's
// thread_index_in_quadgroup. Every kernel writes out[t], t = thread_index_in_threadgroup; compute
// quads are four consecutive threads and simdgroup k is the linear threads [32k, 32k + 32).
// The divergent kernels run the operation only on some lanes. Metal's backend speculates
// `mask_quad_ballot_divergent_k`'s arm, so a lane that skips it still sets its bit in the quad's
// ballot; the lanes `mask_simd_active_divergent_k` skips set no bit of its mask. The expected
// values are derived from t alone in tests/engine_simd.rs and tests/engine_simd_emulated.rs.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

// Bit j of the result is quad lane j, set when that lane runs the ballot with a true predicate.
kernel void mask_quad_ballot_k(device uint *out [[buffer(0)]],
                          uint t [[thread_index_in_threadgroup]]) {
    out[t] = uint(quad_vote::vote_t(quad_ballot((t % 3u) == 0u)));
}

kernel void mask_quad_ballot_divergent_k(device uint *out [[buffer(0)]],
                                    uint t [[thread_index_in_threadgroup]]) {
    uint r = 99u;
    if ((t % 7u) != 2u) {
        r = uint(quad_vote::vote_t(quad_ballot((t % 3u) == 0u)));
    }
    out[t] = r;
}

// Bit i of the 64-bit mask is simdgroup lane i; its upper half is zero, folded in so a nonzero
// upper half shows.
kernel void mask_simd_active_k(device uint *out [[buffer(0)]],
                               uint t [[thread_index_in_threadgroup]]) {
    ulong m = ulong(simd_vote::vote_t(simd_active_threads_mask()));
    out[t] = uint(m) ^ (uint(m >> 32) * 0x9E3779B9u);
}

kernel void mask_simd_active_divergent_k(device uint *out [[buffer(0)]],
                                         uint t [[thread_index_in_threadgroup]]) {
    uint r = 1u;
    if ((t & 31u) >= 3u) {
        ulong m = ulong(simd_vote::vote_t(simd_active_threads_mask()));
        r = uint(m) ^ (uint(m >> 32) * 0x9E3779B9u);
    }
    out[t] = r;
}

struct QuadLaneOut {
    float4 color [[color(0)]];
};

// A fragment's lane in its 2x2 quad: 0 top left, 1 top right, 2 bottom left, 3 bottom right.
fragment QuadLaneOut quad_lane_f(float4 pos [[position]],
                                 ushort q [[thread_index_in_quadgroup]]) {
    QuadLaneOut o;
    o.color = float4(float(q), pos.x, pos.y, 1.0);
    return o;
}
