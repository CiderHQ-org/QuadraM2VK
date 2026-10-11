// The subgroup vote and ballot families, indexed by the lane rather than by the thread.
//
// `air.simd_any` (2 corpus sources), `air.simd_all` (1) and `air.simd_broadcast_first` (3) had no
// authored case; `air.simd_ballot.i64` (6) and `air.simd_is_first` (30) are here because they read
// the same lane model and a case that pinned the votes without the mask they agree with would only
// be half the statement.
//
// Every answer here is a function of `thread_index_in_simdgroup`, and every thread stores at the
// slot its own lane names. The predicate is bit `lane` of a mask supplied in a buffer, so
// `simd_ballot` must reproduce that mask exactly: a lowering whose ballot bits were indexed by the
// physical subgroup lane instead of by `simd_lane_id` returns the mask rotated or truncated, and a
// vote taken over a subgroup wider than the simdgroup mixes in lanes this dispatch never asked
// about. Nothing in the case depends on which thread got which lane, which is what makes it exact.
//
// Three masks, not one. 0xA5A5A5A5 alternates, so `simd_any` is true and `simd_all` is false;
// 0xFFFFFFFF makes both true; 0 makes both false. A lowering that answered `all` as "the ballot is
// non-zero", or `any` as "the ballot is the active mask", agrees with exactly one of the three.
// The high halves of two ballots are stored as well: on a 32-lane simdgroup they must be zero, and
// a 64-bit mask carrying the neighbouring simdgroup's lanes shows up there and nowhere else.
//
// `simd_broadcast_first` is given `lane + 100`, a value that differs in every lane, so its answer
// names which lane it chose: 100 is lane 0 and nothing else is.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_simd_vote_and_ballot(
    device uint *out [[buffer(0)]],
    const device uint *masks [[buffer(1)]],
    uint lane [[thread_index_in_simdgroup]])
{
    bool p0 = ((masks[0] >> lane) & 1u) != 0u;
    bool p1 = ((masks[1] >> lane) & 1u) != 0u;
    bool p2 = ((masks[2] >> lane) & 1u) != 0u;

    simd_vote::vote_t b0 = simd_vote::vote_t(simd_ballot(p0));
    simd_vote::vote_t b1 = simd_vote::vote_t(simd_ballot(p1));
    simd_vote::vote_t b2 = simd_vote::vote_t(simd_ballot(p2));

    device uint *slot = out + lane * 8u;
    slot[0] = uint(b0);
    slot[1] = uint(b0 >> 32);
    slot[2] = uint(b1);
    slot[3] = uint(b2);
    slot[4] = (simd_any(p0) ? 1u : 0u) | (simd_all(p0) ? 2u : 0u)
            | (simd_any(p1) ? 4u : 0u) | (simd_all(p1) ? 8u : 0u)
            | (simd_any(p2) ? 16u : 0u) | (simd_all(p2) ? 32u : 0u)
            | (simd_is_first() ? 64u : 0u);
    slot[5] = simd_broadcast_first(lane + 100u);
    slot[6] = lane;
    slot[7] = uint(b1 >> 32);
}
