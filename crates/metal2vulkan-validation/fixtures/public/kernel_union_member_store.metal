#include <metal_stdlib>
using namespace metal;

// An MSL union lowers to a ONE-MEMBER LLVM struct -- `%union.Slot = type { i64 }` -- so a store
// through a member arrives with a `TypeStruct` pointee that no width-based store rule can answer.
// Every union member starts at byte offset 0, so member 0 of that single-member struct is the
// slot's own address; `emit_single_member_struct_store` descends to it and the ordinary reinterpret
// rules apply. Two members exercise two different rules through the one descent: `f` is a narrowing
// store (read-modify-write, low four bytes only) and `v` is a same-width one (a plain bitcast that
// replaces all eight).
union Slot {
    ulong u;
    float f;
    uint2 v;
};

kernel void kernel_union_member_store(device const uint *in [[buffer(0)]],
                                      device uint *out [[buffer(1)]],
                                      uint tid [[thread_position_in_grid]]) {
    // A dynamic index keeps the union in memory; a scalar local is scalar-replaced by the frontend
    // and the member store never reaches the emitter at all.
    thread Slot s[2];
    uint j = in[tid] & 1u;
    uint other = j ^ 1u;

    s[j].u = 0xdeadbeefcafebabeUL;
    s[other].u = 0x0123456789abcdefUL;

    s[j].f = (float)in[tid] + 0.5f;
    s[other].v = uint2(in[tid] + 3u, 0xfeedfaceu);

    // Read back through an index the frontend cannot relate to `j` or `other`: with the read index
    // taken from a second buffer element, neither member store can be forwarded to the load, so
    // both of them are observed rather than folded away.
    uint r = in[tid + 8u] & 1u;
    out[tid * 2u + 0u] = (uint)(s[r].u & 0xffffffffUL);
    out[tid * 2u + 1u] = (uint)(s[r].u >> 32);
}
