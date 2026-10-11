// A nested loop whose inner body has an early break, a four-way switch, a continue, and whose outer
// loop has a second exit -- the shape that forces the structured planner off its easy path.
//
// SPIR-V needs one merge block per construct, and this CFG does not hand them over: the planner
// rejects twice with `selection:merge-reused` and only admits after splitting the switch tail and
// synthesizing loop-merge and continue blocks. The emitted module carries eleven
// OpLoopMerge/OpSelectionMerge/OpSwitch. Every authored case before this one is a single-function
// module with straight-line or shallow-loop control flow, so the whole merge-assignment and
// block-splitting half of the structurizer is verified by the corpus and by no case.
//
// The `.ll` beside this file is the FRONTEND's lowering of this source (xcrun metal -S -emit-llvm,
// retripled and opaque-pointer-upgraded), not a hand-lowering, so the two halves cannot disagree
// about what the program is.
//
// `in[gid] = gid`, so `v = gid + 7i + 3j` sweeps the low bits across lanes and every switch arm,
// both `break`s and the `continue` fire on some lane. `rounds` is 5 on lanes 2-4 and 6 elsewhere,
// which is the `hits >= 17` outer break being taken or not.
//
//   out[0..7]   = 0x01480DE5 0x00CEF052 0x0003FAA7 0x0007CB4B
//                 0x000472DF 0x0002BE4D 0x008B2E46 0xFD9E10C2   acc
//   out[8..15]  = 20 19 17 17 18 18 18 16                       hits
//   out[16..23] = 6 6 5 5 5 6 6 6                               rounds
//   out[24..31] = 0xCDCDCDCD                                    untouched fill
#include <metal_stdlib>
using namespace metal;

kernel void nested_loop_dispatch(
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
            if ((v & 7u) == 5u) {
                break;
            }
            switch (v & 3u) {
                case 0u: acc += v; break;
                case 1u: acc ^= v; break;
                case 2u: acc = acc * 3u + 1u; break;
                default: acc -= v; break;
            }
            hits += 1u;
            if ((v & 15u) == 9u) {
                continue;
            }
            acc += 2u;
        }
        if (hits >= 17u) {
            break;
        }
        acc = acc * 5u + i;
    }
    out[gid] = acc;
    out[gid + 8] = hits;
    out[gid + 16] = rounds;
}
