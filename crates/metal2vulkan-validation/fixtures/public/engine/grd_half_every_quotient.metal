// Owned fixture: Metal's `half` quotient of every pair of `half` values. A 1024 x 1024 grid: the
// lane at `(c, r)` takes the dividend whose bits are `64 r + c / 16`, divides it by the 4096
// divisors whose bits start at `4096 (c mod 16)`, in four runs of 1024, folds the quotients' bits,
// every NaN as 0x7E00, into an FNV-1a hash, and adds the hash to `out` at the dividend's bits.
// The 16 lanes of a dividend cover every divisor, so the grid covers all 2^32 pairs; a quotient
// that differs changes its lane's hash (each step of the hash is a bijection of the hash before
// it), and so the dividend's sum. The loops' bounds are the constants 4 and 1024.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void grd_half_every_quotient(device atomic_uint *out [[buffer(0)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    uint dividend = 64u * gid.y + (gid.x >> 4);
    half x = as_type<half>(ushort(dividend));
    uint h = 2166136261u;
    for (uint run = 0; run < 4u; ++run) {
        uint first = ((gid.x & 15u) << 12) + (run << 10);
        for (uint i = 0; i < 1024u; ++i) {
            ushort q = as_type<ushort>(x / as_type<half>(ushort(first + i)));
            q = (q & 0x7FFFu) > 0x7C00u ? ushort(0x7E00u) : q;
            h = (h ^ uint(q)) * 16777619u;
        }
    }
    atomic_fetch_add_explicit(&out[dividend], h, memory_order_relaxed);
}
