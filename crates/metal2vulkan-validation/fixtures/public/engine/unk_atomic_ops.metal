// Owned fixture: every texture atomic operation on the textures Metal's
// atomics take, each result read at lane 0 only. Thread t works on texel t of an r32 read_write
// 1D (uint), 1D array (int), 2D array (uint), 3D (int), buffer (uint), and 2D (int, 16-bit
// coordinates) texture, at layer (or slice) t & 1, past the last one from thread 9 on: it
// fetch-adds, -subs, -ands, -ors, -xors, -mins, and -maxes, exchanges, stores, loads, and
// compare-exchanges with a stale and then the current expected texel, and writes each old texel's
// first component (as uint), each compare-exchange's result, and lane 0 of each expected to
// out[40t .. 40t + 39]. Loop-free.
//
// The three `expected` vectors are declared ahead of the switch's only label, so every path into
// the switch bypasses their declarations and Clang emits no lifetime markers for them (as in
// ob2_sized_array.metal): LLVM 22 and later read a marker in a one-argument form, so a fixture
// holding one would decode to a different declaration under each `llvm-as` a gate assembles it
// with. The switch is on t >> 31, zero for every thread.
#include <metal_stdlib>
using namespace metal;

kernel void unk_atomic_ops(texture1d<uint, access::read_write> a1 [[texture(0)]],
                           texture1d_array<int, access::read_write> a1a [[texture(1)]],
                           texture2d_array<uint, access::read_write> a2a [[texture(2)]],
                           texture3d<int, access::read_write> a3 [[texture(3)]],
                           texture_buffer<uint, access::read_write> ab [[texture(4)]],
                           texture2d<int, access::read_write> a2 [[texture(5)]],
                           device uint *out [[buffer(0)]],
                           uint t [[thread_position_in_grid]]) {
    switch (t >> 31) {
        uint4 e;
        int4 f;
        uint4 g;
    default:
        device uint *o = out + 40 * t;
        uint l = t < 9 ? (t & 1) : 2;
        // 1D, uint: every read-modify-write, an exchange, and a load.
        o[0] = a1.atomic_fetch_add(t, uint4(5u)).x;
        o[1] = a1.atomic_fetch_sub(t, uint4(2u)).x;
        o[2] = a1.atomic_fetch_and(t, uint4(0xFF0Fu)).x;
        o[3] = a1.atomic_fetch_or(t, uint4(0x0100u)).x;
        o[4] = a1.atomic_fetch_xor(t, uint4(0x0033u)).x;
        o[5] = a1.atomic_fetch_min(t, uint4(0x1100u + t)).x;
        o[6] = a1.atomic_fetch_max(t, uint4(0x1000u)).x;
        o[7] = a1.atomic_exchange(t, uint4(0xABC0u + t)).x;
        o[8] = a1.atomic_load(t).x;
        // 1D array, int: signed minimum and maximum, an add, an exchange, a store and a load.
        o[9] = as_type<uint>(a1a.atomic_fetch_min(t, l, int4(-3)).x);
        o[10] = as_type<uint>(a1a.atomic_fetch_max(t, l, int4(7)).x);
        o[11] = as_type<uint>(a1a.atomic_fetch_add(t, l, int4(-100)).x);
        o[12] = as_type<uint>(a1a.atomic_exchange(t, l, int4(-77 - int(t))).x);
        a1a.atomic_store(int4(int(t) - 50), t, l);
        o[13] = as_type<uint>(a1a.atomic_load(t, l).x);
        // 2D array, uint: an add, a minimum, a maximum, a stale and a current compare-exchange.
        uint2 c2 = uint2(t, 0);
        o[14] = a2a.atomic_fetch_add(c2, l, uint4(0x10u)).x;
        o[15] = a2a.atomic_fetch_min(c2, l, uint4(0x7000u)).x;
        o[16] = a2a.atomic_fetch_max(c2, l, uint4(0x0040u)).x;
        e = uint4(0xDEADu);
        o[17] = uint(a2a.atomic_compare_exchange_weak(c2, l, &e, uint4(0x1234u)));
        o[18] = e.x;
        o[19] = uint(a2a.atomic_compare_exchange_weak(c2, l, &e, uint4(0x4321u + t)));
        o[20] = e.x;
        o[21] = a2a.atomic_load(c2, l).x;
        // 3D, int: a subtract, and, or, xor, and minimum, and a compare-exchange with the current
        // texel.
        uint3 c3 = uint3(t, 0, l);
        o[22] = as_type<uint>(a3.atomic_fetch_sub(c3, int4(9)).x);
        o[23] = as_type<uint>(a3.atomic_fetch_and(c3, int4(~0x0F)).x);
        o[24] = as_type<uint>(a3.atomic_fetch_or(c3, int4(0x300)).x);
        o[25] = as_type<uint>(a3.atomic_fetch_xor(c3, int4(-1)).x);
        o[26] = as_type<uint>(a3.atomic_fetch_min(c3, int4(-1000)).x);
        f = int4(a3.atomic_load(c3).x);
        o[27] = uint(a3.atomic_compare_exchange_weak(c3, &f, int4(77 + int(t))));
        o[28] = as_type<uint>(f.x);
        o[29] = as_type<uint>(a3.atomic_load(c3).x);
        // Buffer, uint: an add, a maximum, an exchange, a compare-exchange, a store and a load.
        o[30] = ab.atomic_fetch_add(t, uint4(3u)).x;
        o[31] = ab.atomic_fetch_max(t, uint4(0x0800u)).x;
        o[32] = ab.atomic_exchange(t, uint4(0x5000u + t)).x;
        g = uint4(0x5000u + t);
        o[33] = uint(ab.atomic_compare_exchange_weak(t, &g, uint4(0x6000u)));
        o[34] = g.x;
        ab.atomic_store(uint4(0x7000u + t), t);
        o[35] = ab.atomic_load(t).x;
        // 2D, int, 16-bit coordinates: an add, an or, and a load.
        ushort2 s2 = ushort2(t, 0);
        o[36] = as_type<uint>(a2.atomic_fetch_add(s2, int4(11)).x);
        o[37] = as_type<uint>(a2.atomic_fetch_or(s2, int4(0x40)).x);
        o[38] = as_type<uint>(a2.atomic_load(s2).x);
        o[39] = l;
    }
}
