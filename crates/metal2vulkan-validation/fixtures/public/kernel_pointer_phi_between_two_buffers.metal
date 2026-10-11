// A device pointer MERGED ACROSS TWO DESCRIPTORS, twice, with the two merges taking opposite arms.
//
// Metal runs this file; the translator reads the sibling `.ll`, which spells each merge as an
// explicit `phi ptr addrspace(1)` over two predecessor blocks rather than the `select` the Metal
// front end folds this source into. Both compute the same words. A phi is a different emitter path
// from a select -- its arms live in predecessor blocks and only dominate their own edge -- and it is
// the one the corpus modules carrying this shape actually have.
//
// Logical SPIR-V has no value that can be either descriptor, so the merge cannot be a pointer. The
// translator lowers it to a PhysicalStorageBuffer64 DEVICE ADDRESS: each buffer's address is loaded
// from a synthesized address table, the merge selects between the two addresses, and every index
// off the merged pointer becomes a physical access chain. That construction is what this case pins,
// end to end, against what Metal's own pointer does.
//
// `a` holds 100 + i and `b` holds 200 + i, so which arm each merge chose is legible in the output
// word itself. `p` takes `a` on lanes 0-3 and `b` on lanes 4-7; `q` takes the other one; and the
// third row reads `p` again at a DIFFERENT dynamic index, so a merge that is re-derived per use
// rather than carried has to agree with itself. Every expected word is distinct, so a swapped arm,
// a collapsed merge, or a merge that leaks between `p` and `q` moves a value that nothing else
// produces:
//
//   out[0..7]   = 100 101 102 103 204 205 206 207
//   out[8..15]  = 200 201 202 203 104 105 106 107
//   out[16..23] = 107 106 105 104 203 202 201 200
#include <metal_stdlib>
using namespace metal;

kernel void pointer_phi_between_two_buffers(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    device const uint* p = (gid < 4u) ? a : b;
    device const uint* q = (gid < 4u) ? b : a;
    out[gid] = p[gid];
    out[gid + 8] = q[gid];
    out[gid + 16] = p[7u - gid];
}
