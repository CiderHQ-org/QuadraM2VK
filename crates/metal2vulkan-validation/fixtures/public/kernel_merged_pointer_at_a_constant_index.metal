// A device pointer merged across two descriptors, then indexed at a CONSTANT element index.
//
// Metal runs this file; the translator reads the sibling `.ll`, which spells the merge as an
// explicit `phi ptr addrspace(1)` over two predecessor blocks. Both compute the same words.
//
// Logical SPIR-V cannot hold a pointer that is either descriptor, so the merge becomes a
// PhysicalStorageBuffer64 device address and every index off it becomes a physical access chain
// taken from that address. The translator used to do that only when the index was DYNAMIC; a
// constant index fell through to a plain retype, which left an `OpAccessChain` descending into a
// merged pointer whose pointee is a byte -- not a composite -- and the module was refused with
// "owned InBoundsAccessChain violates its access-chain contract".
//
// `a` holds 100 + i and `b` holds 200 + i, so the descriptor each lane's merge chose is legible in
// the word itself. Two DIFFERENT constant indices (1 and 3) are read, so an index folded to zero or
// dropped is visible, and a third row reads the same merge at a dynamic index as the control that
// already worked:
//
//   out[0..7]   = 101 101 101 101 201 201 201 201
//   out[8..15]  = 103 103 103 103 203 203 203 203
//   out[16..23] = 100 101 102 103 204 205 206 207
#include <metal_stdlib>
using namespace metal;

kernel void merged_pointer_at_a_constant_index(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    device const uint* p = (gid < 4u) ? a : b;
    out[gid] = p[1];
    out[gid + 8] = p[3];
    out[gid + 16] = p[gid];
}
