// An integer compare-exchange on a FLOAT device slot, reached through a device pointer that was
// parked in a local struct field and read back by an OUT-OF-LINE helper.
//
// The integer-atomic-over-a-float-slot is what selects the physical-address (device address) model:
// Logical SPIR-V cannot form the integer pointer view of a float slot, because `OpBitcast` may not
// retype a logical pointer. So this fixture executes the whole device-address path for a pointer
// that lives in a local aggregate -- the field store records the pointer as its 64-bit address, the
// helper reads that address back, and the atomic runs on an `OpConvertUToPtr` PhysicalStorageBuffer
// pointer. Twelve corpus sources translate through that path; none of them had an authored case.
//
// Even threads' slots start at 0.0f, whose bits are zero, so their compare-exchange SUCCEEDS and
// overwrites the float with the integer i + 100. Odd threads' slots start at 3.5f, so theirs FAILS
// and leaves the slot alone. `expected` holds the value actually observed either way, so a lost
// write, a write at the wrong address and a read of the wrong memory are all distinguishable.
#include <metal_stdlib>
using namespace metal;

struct cache {
    device float* slots;
    device uint* aux;
};

__attribute__((noinline))
static uint claim(thread const cache& c, uint i)
{
    device atomic_uint* slot = reinterpret_cast<device atomic_uint*>(c.slots + i);
    uint expected = 0u;
    atomic_compare_exchange_weak_explicit(slot, &expected, i + 100u,
                                          memory_order_relaxed, memory_order_relaxed);
    return expected;
}

kernel void kernel_device_address_field_atomic(
    device float* slots [[buffer(0)]],
    device uint* aux [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    cache c;
    c.slots = slots;
    c.aux = aux;
    out[gid] = claim(c, gid);
}
