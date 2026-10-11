// A pointer selected between two descriptors, then selected AGAIN against a third -- a chained
// select whose arm is itself a selected pointer -- and then loaded from and stored through.
//
// Logical SPIR-V cannot hold a pointer value that is any of three descriptors, so the emitter
// cannot merge them. It builds a SELECTED ACCESS TREE instead: the merge is kept as the conditions
// that produced it, and every consumer is replayed once per leaf under the branch condition that
// selects it. A load becomes a value select over the arms' loads; a store becomes a
// read-modify-write on every arm, so that each lane writes the new value into the descriptor its
// merge chose and writes each other descriptor's own bytes back unchanged.
//
// That replay is what this case pins, and the store row is what makes it observable. `a` holds
// 100 + i, `b` 200 + i and `c` 300 + i. `p` is `a` on odd lanes and `b` on even ones; `q` is `p` on
// lanes 0-3 and `c` on lanes 4-7. Lane `gid` then writes `500 + gid` into slot `gid + 8` of
// whichever buffer `q` names -- a slot no other lane touches, so there is no race -- and reads slot
// `gid + 8` of ALL THREE buffers back. Exactly one of the last three rows may show `500 + gid` on
// each lane; the other two must still hold their initial bytes, which is the read-modify-write
// contract stated as an output.
//
//   out[0..7]   = 200 101 202 103 304 305 306 307   q[gid]
//   out[8..15]  = 201 101 201 101 301 301 301 301   q[1], a constant index off the merge
//   out[16..23] = 108 501 110 503 112 113 114 115   a[gid+8]
//   out[24..31] = 500 209 502 211 212 213 214 215   b[gid+8]
//   out[32..39] = 308 309 310 311 504 505 506 507   c[gid+8]
#include <metal_stdlib>
using namespace metal;

kernel void chained_pointer_selects(
    device uint* a [[buffer(0)]],
    device uint* b [[buffer(1)]],
    device uint* c [[buffer(2)]],
    device uint* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    device uint* p = (gid & 1u) ? a : b;
    device uint* q = (gid < 4u) ? p : c;
    out[gid] = q[gid];
    out[gid + 8] = q[1];
    q[gid + 8] = gid + 500u;
    out[gid + 16] = a[gid + 8];
    out[gid + 24] = b[gid + 8];
    out[gid + 32] = c[gid + 8];
}
