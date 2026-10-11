// A loop-carried phi of DEVICE POINTERS whose backedge arm is a pointer loaded out of device
// memory -- the shape `emit_bda_address_phi` exists for, and the one no authored case reached.
//
// `struct Node { device uint *next; uint value; }` makes the frontend emit `root` as an
// `air.indirect_buffer` with a nested `air.buffer` member, so `cur->next` is a real
// `load ptr addrspace(1), ptr addrspace(1)`. At the loop header `cur` is a phi of the entry
// argument and that load, and the load is BEHIND the header, so `bda_phi_address_id` has to
// reserve the address of a pointer it has not lowered yet. Reserving the wrong id there emitted an
// `OpIAdd %ulong` over an operand nothing ever defined; `6f20d398` is the fix, and this is the
// device evidence for it.
//
// The walk is real and it terminates. `root.next` points at a second node whose own `next` is
// null, and the null guard keeps `cur` on the last valid node instead of dereferencing zero, so
// with `steps = 2` the sum visits node 0 (7) then node 1 (11) and the trailing read adds node 1
// again: 7 + 11 + 11 = 29, the same on both backends and independent of thread id.
#include <metal_stdlib>
using namespace metal;

struct Node {
    device uint *next;
    uint value;
};

kernel void kernel_device_pointer_chain_phi(
    device Node *root [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant uint &steps [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    device Node *cur = root;
    uint sum = 0u;
    for (uint i = 0u; i < steps; ++i) {
        sum += cur->value;
        device Node *nxt = reinterpret_cast<device Node *>(cur->next);
        cur = (nxt == nullptr) ? cur : nxt;
    }
    out[gid] = sum + cur->value;
}
