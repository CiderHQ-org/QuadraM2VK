#include <metal_stdlib>
using namespace metal;

kernel void kernel_bitfield_and_wide_counts(
    device uint *out [[buffer(0)]],
    device const uint *u [[buffer(1)]],
    device const ulong *w [[buffer(2)]])
{
    uint base = u[0], ins = u[1], off = u[2], bits = u[3];
    int sbase = as_type<int>(u[4]);

    uint a = insert_bits(base, ins, off, bits);
    int  b = insert_bits(sbase, as_type<int>(ins), off, bits);
    ulong c = insert_bits(w[0], w[1], u[5], u[6]);

    int  d = extract_bits(sbase, u[7], u[8]);
    ulong e = extract_bits(w[2], u[9], u[10]);

    ulong2 counts = popcount(ulong2(w[3], w[4]));

    out[0] = a;
    out[1] = as_type<uint>(b);
    out[2] = uint(c & 0xffffffffUL);  out[3] = uint(c >> 32);
    out[4] = as_type<uint>(d);
    out[5] = uint(e & 0xffffffffUL);  out[6] = uint(e >> 32);
    out[7] = uint(counts.x);          out[8] = uint(counts.y);
    out[9]  = uint(clz(w[5]));
    out[10] = uint(ctz(w[5]));
    out[11] = uint(clz(w[6]));
    out[12] = uint(ctz(w[6]));
}
