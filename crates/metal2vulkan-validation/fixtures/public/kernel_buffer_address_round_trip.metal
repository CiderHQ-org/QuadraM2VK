#include <metal_stdlib>
using namespace metal;

struct Row {
    ushort a;
    ushort b;
    ushort c;
};

struct Params {
    uint byte_offset;
};

// The structuring-element read shape Apple's DilateFilter4/ErodeFilter4 use: take the integer
// address of a bound buffer, add a byte offset the host supplies, and read the result back as a
// pointer. Every load lands inside `base`; the address itself is never observed.
kernel void buffer_address_round_trip(
    constant Row* base [[buffer(0)]],
    constant Params& p [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    constant Row* row = reinterpret_cast<constant Row*>(reinterpret_cast<uintptr_t>(base) + p.byte_offset);
    Row r = row[gid];
    out[gid] = uint(r.a) | (uint(r.b) << 8) | (uint(r.c) << 16);
}
