// Owned fixture: simdgroup and quad operations in compute kernels.
// Every kernel writes out[t], t = thread_index_in_threadgroup; lane = thread_index_in_simdgroup
// (t & 31: Metal simdgroup k is the linear threads [32k, 32k + 32)). Divergent kernels run the
// operation only on some lanes, and inactive lanes contribute nothing, except where Metal's
// backend speculates the branch's arm: `quad_ballot_k`'s quad votes count every lane of the quad.
// The expected values are derived from t alone in tests/engine_simd.rs. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void simd_shuffle_k(device uint *out [[buffer(0)]],
                           uint t [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    out[t] = simd_shuffle(t, ushort((lane * 7u + 3u) & 31u));
}

kernel void simd_broadcast_k(device uint *out [[buffer(0)]],
                             uint t [[thread_index_in_threadgroup]]) {
    out[t] = simd_broadcast(t, ushort(9));
}

kernel void simd_shuffle_up_k(device uint *out [[buffer(0)]],
                              uint t [[thread_index_in_threadgroup]]) {
    out[t] = simd_shuffle_up(t, ushort(3));
}

kernel void simd_shuffle_down_k(device uint *out [[buffer(0)]],
                                uint t [[thread_index_in_threadgroup]]) {
    out[t] = simd_shuffle_down(t, ushort(5));
}

kernel void simd_shuffle_xor_k(device uint *out [[buffer(0)]],
                               uint t [[thread_index_in_threadgroup]]) {
    out[t] = simd_shuffle_xor(t, ushort(13));
}

kernel void simd_rotate_k(device uint *out [[buffer(0)]],
                          uint t [[thread_index_in_threadgroup]]) {
    out[t] = simd_shuffle_rotate_down(t, ushort(7));
}

kernel void simd_fill_down_k(device uint *out [[buffer(0)]],
                             uint t [[thread_index_in_threadgroup]]) {
    out[t] = simd_shuffle_and_fill_down(t, t + 1000u, ushort(3), ushort(8));
}

kernel void simd_fill_up_k(device uint *out [[buffer(0)]],
                           uint t [[thread_index_in_threadgroup]]) {
    out[t] = simd_shuffle_and_fill_up(t, t + 1000u, ushort(2));
}

kernel void simd_sum_k(device uint *out [[buffer(0)]],
                       uint t [[thread_index_in_threadgroup]]) {
    out[t] = simd_sum(t) + uint(simd_sum(float(t & 7u))) * 100000u;
}

kernel void simd_half_sum_k(device uint *out [[buffer(0)]],
                            uint t [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    out[t] = uint(float(simd_sum(half(lane))));
}

kernel void simd_minmax_k(device uint *out [[buffer(0)]],
                          uint t [[thread_index_in_threadgroup]]) {
    short v = short(t & 31u) - short(20);
    out[t] = uint(int(simd_min(v)) + 1000) + uint(int(simd_max(v)) + 1000) * 10000u;
}

kernel void simd_bits_k(device uint *out [[buffer(0)]],
                        uint t [[thread_index_in_threadgroup]]) {
    ushort v = ushort(1u << (t & 15u));
    out[t] = uint(simd_or(v)) | (uint(simd_and(ushort(t | 0xF0u))) << 16);
}

kernel void simd_xor_k(device uint *out [[buffer(0)]],
                       uint t [[thread_index_in_threadgroup]]) {
    out[t] = uint(simd_xor(ushort(t * t)));
}

kernel void simd_prefix_k(device uint *out [[buffer(0)]],
                          uint t [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    if (lane % 3u != 0u) {
        out[t] = simd_prefix_inclusive_sum(t) * 2u + simd_prefix_exclusive_sum(lane) * 100000u;
    } else {
        out[t] = 7u;
    }
}

kernel void simd_float_prefix_k(device uint *out [[buffer(0)]],
                                uint t [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    if (lane >= 4u) {
        out[t] = uint(simd_prefix_exclusive_sum(float(lane)));
    } else {
        out[t] = 7u;
    }
}

kernel void simd_vote_k(device uint *out [[buffer(0)]],
                        uint t [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint r = 0u;
    if (lane != 6u) {
        r = uint(simd_all(lane != 6u)) + uint(simd_any(t == 40u)) * 2u + uint(simd_all(t < 40u)) * 4u;
    }
    out[t] = r;
}

kernel void simd_ballot_k(device uint *out [[buffer(0)]],
                          uint t [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint r = 1u;
    if (lane >= 2u) {
        ulong b = ulong(simd_ballot(t % 3u == 0u));
        r = uint(b) ^ uint(b >> 32);
    }
    out[t] = r;
}

kernel void simd_first_k(device uint *out [[buffer(0)]],
                         uint t [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint r = 3u;
    if (lane >= 5u) {
        r = uint(simd_is_first()) + simd_broadcast_first(t) * 10u;
    }
    out[t] = r;
}

kernel void quad_ops_k(device uint *out [[buffer(0)]],
                       uint t [[thread_index_in_threadgroup]]) {
    out[t] = quad_shuffle_xor(t, ushort(1)) + quad_broadcast(t, ushort(2)) * 1000u
        + quad_sum(t) * 1000000u;
}

kernel void quad_relative_k(device uint *out [[buffer(0)]],
                            uint t [[thread_index_in_threadgroup]]) {
    out[t] = quad_shuffle_up(t, ushort(1)) + quad_shuffle_down(t, ushort(2)) * 1000u
        + quad_shuffle_rotate_down(t, ushort(3)) * 1000000u;
}

kernel void quad_dynamic_k(device uint *out [[buffer(0)]],
                           uint t [[thread_index_in_threadgroup]]) {
    out[t] = quad_shuffle_xor(t, ushort(t >> 2) & ushort(3)) + quad_max(t) * 1000u;
}

kernel void quad_ballot_k(device uint *out [[buffer(0)]],
                          uint t [[thread_index_in_threadgroup]]) {
    uint r = 99u;
    if ((t % 7u) != 2u) {
        r = uint(quad_is_first()) + uint(quad_vote::vote_t(quad_active_threads_mask())) * 2u
            + uint(quad_all(t != 9u)) * 64u + uint(quad_any(t == 9u)) * 128u;
    }
    out[t] = r;
}
