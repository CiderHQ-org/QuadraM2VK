// Owned fixture (written for this repository): `quad_is_helper_thread` and
// `simd_is_helper_thread` in a kernel, a vertex function and fragment functions, with and without
// a discard.
// - A kernel thread or a vertex writes its own answers, q | s << 1, with 0x100 set, and the
//   kernel its quad's and simdgroup's votes on them: no kernel or vertex thread is a helper.
// - Fragment functions are drawn over the whole target. Over 8x8 no pixel's lane is a helper; over
//   7x5 the last column and row are quads the target cuts, whose lanes past the target Metal runs
//   as helpers. A pixel returns its own answers, its quad's ballots of each, and
//   quad_active_threads_mask: the helpers are active lanes of their quad.
// - A discard of the pixels with (x + y) mod 3 == 0 makes each a helper. Every quad keeps two or
//   three live pixels, which see the discarded ones as helpers after the discard and not before.
// - `fragment_helper_threads_beside_a_simdgroup_vote` also votes over its simdgroup after the
//   discard, so a discarded thread is a lane of a simdgroup operation.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void helper_threads_in_a_kernel(device uint *out [[buffer(1)]],
                                       uint t [[thread_position_in_grid]])
{
    const bool q = quad_is_helper_thread();
    const bool s = simd_is_helper_thread();
    out[t] = uint(q) | uint(s) << 1 | uint(quad_any(q)) << 2 | uint(simd_any(s)) << 3 | 0x100u;
}

vertex void helper_threads_in_a_vertex_function(uint vid [[vertex_id]],
                                                device uint *out [[buffer(1)]])
{
    out[vid] = uint(quad_is_helper_thread()) | uint(simd_is_helper_thread()) << 1 | 0x100u;
}

static uint bits(quad_vote v)
{
    return uint(quad_vote::vote_t(v));
}

static uint4 answers(bool q, bool s)
{
    return uint4(uint(q) | uint(s) << 1, bits(quad_ballot(q)), bits(quad_ballot(s)),
                 bits(quad_active_threads_mask()));
}

static bool discarded(float4 position)
{
    const uint2 p = uint2(position.xy);
    return (p.x + p.y) % 3u == 0u;
}

fragment uint4 fragment_helper_threads(float4 position [[position]])
{
    return answers(quad_is_helper_thread(), simd_is_helper_thread());
}

fragment uint4 fragment_helper_threads_after_a_discard(float4 position [[position]])
{
    if (discarded(position)) {
        discard_fragment();
    }
    return answers(quad_is_helper_thread(), simd_is_helper_thread());
}

fragment uint4 fragment_helper_threads_before_a_discard(float4 position [[position]])
{
    const bool q = quad_is_helper_thread();
    const bool s = simd_is_helper_thread();
    if (discarded(position)) {
        discard_fragment();
    }
    return answers(q, s);
}

fragment uint4 fragment_helper_threads_beside_a_simdgroup_vote(float4 position [[position]])
{
    if (discarded(position)) {
        discard_fragment();
    }
    const bool q = quad_is_helper_thread();
    const bool s = simd_is_helper_thread();
    uint4 r = answers(q, s);
    r.x |= uint(simd_any(s)) << 2;
    return r;
}
