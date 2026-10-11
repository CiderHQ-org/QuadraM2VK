// A CONTENDED integer atomic on a FLOAT device slot, reached through a device pointer parked in a
// local struct field and read back by an out-of-line helper.
//
// The integer-atomic-over-a-float-slot is what selects the physical-address (device address) model:
// Logical SPIR-V cannot form the integer pointer view of a float slot, because `OpBitcast` may not
// retype a logical pointer. `kernel_device_address_contended_atomic` differs from
// `kernel_device_address_field_atomic` in the two ways that decide an atomic's SCOPE: the operation
// is a read-modify-write rather than a compare-exchange, and every threadgroup in the grid targets
// the SAME four words, so an atomic that is atomic only within one threadgroup loses updates.
//
// Each of the 256 threads adds 1 to the slot named by `gid % 4`. A contended add's per-thread
// RESULT is order-dependent, so it is not compared; the four accumulators are exact. Each must end
// at 64, which separates a lost update, an add at the wrong address, and an add that never reached
// device memory. The slots start at 0.0f, whose bits are zero, so the final bits ARE the count.
#include <metal_stdlib>
using namespace metal;

struct cache {
    device float* slots;
    device uint* seen;
};

__attribute__((noinline))
static void bump(thread const cache& c, uint slot)
{
    device atomic_uint* cell = reinterpret_cast<device atomic_uint*>(c.slots + slot);
    atomic_fetch_add_explicit(cell, 1u, memory_order_relaxed);
}

kernel void kernel_device_address_contended_atomic(
    device float* slots [[buffer(0)]],
    device uint* seen [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    cache c;
    c.slots = slots;
    c.seen = seen;
    uint slot = gid & 3u;
    bump(c, slot);
    seen[gid] = slot;
}
