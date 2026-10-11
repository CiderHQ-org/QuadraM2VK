// A subword vector read from, and written back through, a WORD pointer -- the `*(device ushort2*)`
// reinterpret that Metal code writes whenever it treats a `uint` buffer as pairs of halves.
//
// In SPIR-V the pointer keeps its declared `uint` pointee, so both directions are equal-width
// reinterprets of the VALUE: the load is `OpLoad` of the word plus `OpBitcast` up to `ushort2`, and
// the store is `OpBitcast` down to the word plus `OpStore`. They are the same two instructions in
// mirror image, and the emitter has to accept the same pointee/object pairs on both sides -- a
// store arm narrower than the load arm refuses to write back what the load just read.
//
// `in[i]` packs two DIFFERENT halves, `0x0100 + i` low and `0x0200 + i` high, so a swapped lane
// pair is visible as a value and not just as a width. The first two rows extract each half through
// ordinary word stores, which pins which half is which; the third row is the reinterpret store
// itself, writing the halves back SWAPPED as one word. The fourth row is never written and must
// still hold its fill: a store that took the narrowing path instead would leave half of
// `out[16 + i]` at the fill byte, and a store that overshot its word would land in row three.
//
//   out[0..7]   = 0x0100 + i                      low half, word store
//   out[8..15]  = 0x0200 + i                      high half, word store
//   out[16..23] = ((0x0100 + i) << 16) | (0x0200 + i)   halves swapped, reinterpret store
//   out[24..31] = 0xCDCDCDCD                      untouched fill
#include <metal_stdlib>
using namespace metal;

kernel void subword_reinterpret_roundtrip(
    device const uint* in [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    ushort2 v = *(device const ushort2*)&in[gid];
    out[gid] = uint(v.x);
    out[gid + 8] = uint(v.y);
    *(device ushort2*)&out[gid + 16] = ushort2(v.y, v.x);
}
