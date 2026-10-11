// Three loop shapes in one kernel, chosen so the structured planner has to synthesize merges
// rather than find them: a nested loop whose inner body has an early break, a four-way switch and a
// continue and whose outer loop has a second exit; a do-while (bottom-test latch) with two breaks
// out of it; and a switch with SHARED case continuations inside a third loop, one of whose arms
// forces the induction variable past its bound.
//
// Measured with llvm-cov over all 959 corpus-backed cased sources plus all 126 other public
// fixtures, this one module reaches TWELVE production functions no case and no other fixture
// reaches: `rewrite_switch_bypass_merge` (native/cfg/blocks.rs); `synth_multi_exit_merge`,
// `synth_two_target_dispatch`, `synth_multi_latch_continue`, `split_multi_exit_critical_edges` and
// `split_two_target_critical_edges` (structured_emit/multi_exit.rs); `arm_defs_safe_to_clone`,
// `privatize_trivial` and `terminator_mentions` (cfg/clone_crossarm/cross_arm.rs);
// `privatize_direct_construct_tree_shared_continuations` (structured_emit/straddle.rs);
// `duplicate_phi_incoming` (tir/phi_edit.rs); and `rename_switch` (tir/rename.rs). That is the
// multi-exit funnel, the cross-arm privatizer and the switch-tail splitter -- the half of the
// structurizer the corpus exercises and no authored case did.
//
// The `.ll` beside this file is the FRONTEND's lowering of this source (xcrun metal -S -emit-llvm,
// retripled and opaque-pointer-upgraded), not a hand-lowering, so the two halves cannot disagree
// about what the program is.
//
// `in[gid] = gid`, so `v = gid + 7i + 3j` sweeps the low bits across lanes and every switch arm,
// every break and the continue fire on some lane: `rounds` is 5 on lanes 2-4 and 6 elsewhere (the
// `hits >= 17` outer break), and `steps` is 10 on lanes 0,1,4,5 and 1 on the rest (the two
// do-while exits). A merge assigned to the wrong construct moves one of those counters as well as
// the accumulators.
//
//   out[0..7]   = 0x014811D7 0x00CEF444 0x0003FAA5 0x0007CB49
//                 0x000476D1 0x0002C23F 0x008B2E42 0xFD9E10C6   acc
//   out[8..15]  = 20 19 17 17 18 18 18 16                       hits
//   out[16..23] = 6 6 5 5 5 6 6 6                               rounds
//   out[24..31] = 10 10 1 1 10 10 1 1                           steps
//   out[32..39] = 0x12 0x3B 0x58 0x8D 0xC8 0x43 0x65 0xD9       tally
//   out[40..47] = 0xCDCDCDCD                                    untouched fill
#include <metal_stdlib>
using namespace metal;

kernel void three_loop_shapes(
    device const uint* in [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint seed = in[gid];
    uint acc = 0u;
    uint hits = 0u;
    uint rounds = 0u;
    for (uint i = 0u; i < 6u; ++i) {
        uint row = seed + i * 7u;
        rounds += 1u;
        for (uint j = 0u; j < 5u; ++j) {
            uint v = row + j * 3u;
            if ((v & 7u) == 5u) { break; }
            switch (v & 3u) {
                case 0u: acc += v; break;
                case 1u: acc ^= v; break;
                case 2u: acc = acc * 3u + 1u; break;
                default: acc -= v; break;
            }
            hits += 1u;
            if ((v & 15u) == 9u) { continue; }
            acc += 2u;
        }
        if (hits >= 17u) { break; }
        acc = acc * 5u + i;
    }

    // do-while (bottom-test latch) with a three-way exit
    uint state = seed | 1u;
    uint steps = 0u;
    do {
        steps += 1u;
        if ((state & 3u) == 3u) { acc ^= state; break; }
        if (steps > 9u) { acc += 1000u; break; }
        state = state * 3u + 1u;
    } while ((state & 0xFFu) != 0u);
    acc += steps;

    // switch with shared case continuations inside a loop
    uint tally = 0u;
    for (uint k = 0u; k < 7u; ++k) {
        uint w = seed * 3u + k;
        switch (w % 5u) {
            case 0u:
            case 2u:
                tally += w;
                break;
            case 1u:
                tally ^= w;
                if ((w & 1u) != 0u) { continue; }
                break;
            case 3u:
                tally = tally * 3u + 1u;
                break;
            default:
                if (tally > 4000u) { k = 7u; }
                tally += 5u;
                break;
        }
        tally += 1u;
    }

    out[gid] = acc;
    out[gid + 8] = hits;
    out[gid + 16] = rounds;
    out[gid + 24] = steps;
    out[gid + 32] = tally;
}
