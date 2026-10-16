// metal-std: macos-metal2.2
// Owned fixture (written for this repository): `quad_is_helper_thread` and
// `simd_is_helper_thread` after `discard_fragment` with terminate semantics (before MSL 2.3),
// compiled with the standard named on the first line. Drawn over the whole target; the pixels
// with (x + y) mod 3 == 0 discard, and every quad keeps two or three live pixels.
// - `fragment_helper_threads_after_a_terminating_discard`: quad ballots follow the discard, so a
//   discarded thread keeps running for them; Metal runs it as a helper, which its quad's live
//   pixels see, as at MSL 2.3.
// - `fragment_helper_thread_alone_after_a_terminating_discard`: only the pixel's own answers
//   follow, so nothing can tell the discard from the thread's end; a live pixel is no helper.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

static uint bits(quad_vote v)
{
    return uint(quad_vote::vote_t(v));
}

static bool discarded(float4 position)
{
    const uint2 p = uint2(position.xy);
    return (p.x + p.y) % 3u == 0u;
}

fragment uint4 fragment_helper_threads_after_a_terminating_discard(float4 position [[position]])
{
    if (discarded(position)) {
        discard_fragment();
    }
    const bool q = quad_is_helper_thread();
    const bool s = simd_is_helper_thread();
    return uint4(uint(q) | uint(s) << 1, bits(quad_ballot(q)), bits(quad_ballot(s)),
                 bits(quad_active_threads_mask()));
}

fragment uint4 fragment_helper_thread_alone_after_a_terminating_discard(float4 position [[position]])
{
    if (discarded(position)) {
        discard_fragment();
    }
    return uint4(uint(quad_is_helper_thread()) | uint(simd_is_helper_thread()) << 1, 7u, 7u, 7u);
}
