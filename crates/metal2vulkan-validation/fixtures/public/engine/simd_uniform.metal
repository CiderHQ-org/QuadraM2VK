// Owned fixture: simdgroup and quad operations in workgroup-uniform
// control flow, which a kernel whose simdgroups are emulated through workgroup memory runs
// exactly. Every kernel writes out[t], t = thread_index_in_threadgroup; lane =
// thread_index_in_simdgroup (t & 31: Metal simdgroup k is the linear threads [32k, 32k + 32)).
// The expected values are derived from t and the threadgroup size in
// tests/engine_simd_emulated.rs, including partially occupied simdgroups. `divergent_k` sums
// on some lanes only, which the emulation refuses. `region_k` moves eight words per lane and reads
// a `[[threadgroup(0)]]` region, so its arena is declared even when the region is empty.
#include <metal_stdlib>
using namespace metal;

kernel void vote_k(device uint *out [[buffer(0)]],
                   uint t [[thread_index_in_threadgroup]]) {
    out[t] = uint(simd_all(t != 40u)) + uint(simd_any(t == 40u)) * 2u
        + uint(simd_is_first()) * 4u + simd_broadcast_first(t) * 8u;
}

kernel void ballot_k(device uint *out [[buffer(0)]],
                     uint t [[thread_index_in_threadgroup]]) {
    ulong b = ulong(simd_ballot(t % 3u == 0u));
    out[t] = uint(b) ^ uint(b >> 32);
}

kernel void prefix_k(device uint *out [[buffer(0)]],
                     uint t [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    out[t] = simd_prefix_inclusive_sum(t) * 2u + simd_prefix_exclusive_sum(lane) * 100000u;
}

kernel void float_prefix_k(device uint *out [[buffer(0)]],
                           uint t [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    out[t] = uint(simd_prefix_exclusive_sum(float(lane)))
        + uint(float(simd_prefix_inclusive_sum(half(t & 7u)))) * 10000u;
}

kernel void quad_vote_k(device uint *out [[buffer(0)]],
                        uint t [[thread_index_in_threadgroup]]) {
    out[t] = uint(quad_is_first()) + uint(quad_vote::vote_t(quad_active_threads_mask())) * 2u
        + uint(quad_all(t != 9u)) * 64u + uint(quad_any(t == 9u)) * 128u
        + quad_min(t) * 256u;
}

kernel void vector_k(device uint *out [[buffer(0)]],
                     uint t [[thread_index_in_threadgroup]]) {
    float4 v = float4(float(t), float(t + 1u), float(t ^ 0x55u), float(255u - t));
    uint4 s = uint4(simd_shuffle_xor(v, ushort(5)));
    out[t] = s.x | (s.y << 8) | (s.z << 16) | (s.w << 24);
}

kernel void half_vector_k(device uint *out [[buffer(0)]],
                          uint t [[thread_index_in_threadgroup]]) {
    half2 h = simd_shuffle_down(half2(half(t & 63u), half(t >> 1)), ushort(3));
    out[t] = uint(float(h.x)) | (uint(float(h.y)) << 16);
}

kernel void loop_k(device uint *out [[buffer(0)]],
                   constant uint &steps [[buffer(1)]],
                   uint t [[thread_index_in_threadgroup]]) {
    uint v = t;
    for (uint i = 0u; i < steps; i++) {
        v = simd_shuffle_rotate_down(v, ushort(1)) + i;
    }
    out[t] = v;
}

kernel void barrier_k(device uint *out [[buffer(0)]],
                      uint t [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint3 size [[threads_per_threadgroup]]) {
    threadgroup uint tg[128];
    tg[t] = t * 3u;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint src = (sg << 5) | ((lane + 1u) & 31u);
    out[t] = src < size.x * size.y * size.z ? tg[src] : 0u;
}

kernel void divergent_k(device uint *out [[buffer(0)]],
                        uint t [[thread_index_in_threadgroup]]) {
    uint r = 0u;
    if (t < 16u) {
        r = simd_sum(t);
    }
    out[t] = r;
}

kernel void region_k(device uint *out [[buffer(0)]],
                     threadgroup uint *tg [[threadgroup(0)]],
                     uint t [[thread_index_in_threadgroup]]) {
    uint4 v = simd_shuffle_and_fill_down(uint4(t, t + 1u, t + 2u, t + 3u), uint4(t << 8),
                                         ushort(1));
    if (t == 0u) {
        tg[0] = 5u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[t] = v.x + v.y + v.z + v.w + tg[0];
}
