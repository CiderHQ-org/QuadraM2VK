// Owned fixture: `frexp` and `ilogb`, which Apple's compiler
// emits as calls of runtime routines its backend defines at pipeline creation. Each thread reads
// one input word and writes eight: the scalar forms, and lanes of the 4-lane forms beside other
// lanes' constants. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

// Input word `i` is a `float`'s bits. Output words 8i to 8i + 7: frexp(x)'s mantissa bits and
// exponent, ilogb(x), precise::ilogb(x), then of float4(0.25, x, -7, x): frexp's lane 1 mantissa
// bits, lane 3 exponent, ilogb's lane 1, and lane 0's mantissa bits ^ lane 2's exponent ^ lane 2's
// ilogb ^ the bit count of the input word times 2^32 + 1 (twice its own) at bit 16: a 64-bit
// operation, so a declaration of a 64-bit type precedes the lifetime markers' (the decoded
// printout is then the same under every LLVM major).
kernel void frexp_ilogb_float(device const uint *a [[buffer(0)]],
                              device uint *o [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(a[i]);
    int e;
    float m = frexp(x, e);
    float4 v = float4(0.25f, x, -7.0f, x);
    int4 e4;
    float4 m4 = frexp(v, e4);
    int4 l4 = ilogb(v);
    o[8 * i + 0] = as_type<uint>(m);
    o[8 * i + 1] = uint(e);
    o[8 * i + 2] = uint(ilogb(x));
    o[8 * i + 3] = uint(precise::ilogb(x));
    o[8 * i + 4] = as_type<uint>(m4.y);
    o[8 * i + 5] = uint(e4.w);
    o[8 * i + 6] = uint(l4.y);
    o[8 * i + 7] = as_type<uint>(m4.x) ^ uint(e4.z) ^ uint(l4.z) ^
                   (uint(popcount(ulong(a[i]) * 0x100000001ul)) << 16);
}

// The same for `half`: input word `i`'s low 16 bits are a `half`'s, and each mantissa is written
// as its 16 bits.
kernel void frexp_ilogb_half(device const uint *a [[buffer(0)]],
                             device uint *o [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    half x = as_type<half>(ushort(a[i]));
    int e;
    half m = frexp(x, e);
    half4 v = half4(0.25h, x, -7.0h, x);
    int4 e4;
    half4 m4 = frexp(v, e4);
    int4 l4 = ilogb(v);
    o[8 * i + 0] = uint(as_type<ushort>(m));
    o[8 * i + 1] = uint(e);
    o[8 * i + 2] = uint(ilogb(x));
    o[8 * i + 3] = uint(precise::ilogb(x));
    o[8 * i + 4] = uint(as_type<ushort>(m4.y));
    o[8 * i + 5] = uint(e4.w);
    o[8 * i + 6] = uint(l4.y);
    o[8 * i + 7] = uint(as_type<ushort>(m4.x)) ^ uint(e4.z) ^ uint(l4.z) ^
                   (uint(popcount(ulong(a[i]) * 0x100000001ul)) << 16);
}
