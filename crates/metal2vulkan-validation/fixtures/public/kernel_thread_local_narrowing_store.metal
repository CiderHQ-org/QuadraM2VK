#include <metal_stdlib>
using namespace metal;

// A thread-local `ulong` slot written through a `float` pointer: a store of a NARROWER scalar than
// the slot's declared pointee. Logical SPIR-V has no partial store, so the emitter turns it into a
// read-modify-write that changes only the slot's low four bytes -- and this case is what checks
// that the high four bytes really are left alone.
kernel void kernel_thread_local_narrowing_store(device const uint *in [[buffer(0)]],
                                                device uint *out [[buffer(1)]],
                                                uint tid [[thread_position_in_grid]]) {
    // Two slots and a DYNAMIC index. A scalar local is scalar-replaced by the frontend and the
    // reinterpret never reaches the emitter; the dynamic index keeps the array in memory.
    thread ulong s[2];
    uint j = in[tid] & 1u;
    uint other = j ^ 1u;

    s[j] = 0xdeadbeefcafebabeUL;
    s[other] = 0x0123456789abcdefUL;
    *(thread float *)&s[j] = (float)in[tid] + 0.5f;

    out[tid * 3u + 0u] = (uint)(s[j] & 0xffffffffUL);   // the narrowed store's own bytes
    out[tid * 3u + 1u] = (uint)(s[j] >> 32);            // the bytes it must NOT have touched
    out[tid * 3u + 2u] = (uint)(s[other] & 0xffffffffUL); // the slot it must not have reached
}
