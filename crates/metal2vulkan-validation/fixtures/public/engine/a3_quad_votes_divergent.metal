// Owned fixture: quad votes under divergence: quad_is_first, the
// active-lane mask, quad_all and quad_any (not quad_ballot, which no corpus row calls and the
// decoder does not model). Compute quads are four consecutive threads. Lane t & 3 of quad
// q = t >> 2 runs the votes exactly when bit (t & 3) of q & 15 is set, so the sixteen quads of 64
// threads cover every subset of a quad once, including the empty one and those whose first lane
// is inactive. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a3_quad_votes_divergent(device uint *out [[buffer(0)]],
                                    uint t [[thread_index_in_threadgroup]]) {
    uint q = t >> 2;
    uint r = 0xFF000000u;
    if ((((q & 15u) >> (t & 3u)) & 1u) != 0u) {
        r = uint(quad_is_first())
            | (uint(quad_vote::vote_t(quad_active_threads_mask())) << 1)
            | (uint(quad_all((t & 1u) == 0u)) << 5)
            | (uint(quad_any((t % 5u) == 0u)) << 6)
            | (uint(quad_all(t != 9u)) << 7)
            | (uint(quad_any(t == 9u)) << 8);
    }
    out[t] = r;
}
