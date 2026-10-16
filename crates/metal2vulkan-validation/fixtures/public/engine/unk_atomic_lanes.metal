// Owned fixture: the probe of what Metal's texture atomics return in every
// lane of their `vec<T, 4>` results, and what a compare-exchange leaves in every lane of
// `expected`. Thread t works on texel (t, 0) of an r32uint and an r32sint read_write 2D texture:
// it fetch-adds, fetch-ors, exchanges, loads, compare-exchanges with a stale and then the current
// value, fetch-mins the int texel, stores and loads back, and exchanges the int texel, keeping
// all four lanes of every result and of each `expected` in out[10t .. 10t + 9]. Loop-free.
//
// Both `expected` vectors are declared ahead of the switch's only label, so every path into the
// switch bypasses their declarations and Clang emits no lifetime markers for them (as in
// ob2_sized_array.metal): LLVM 22 and later read a marker in a one-argument form, so a fixture
// holding one would decode to a different declaration under each `llvm-as` a gate assembles it
// with. The switch is on t >> 31, zero for every thread.
#include <metal_stdlib>
using namespace metal;

kernel void unk_atomic_lanes(texture2d<uint, access::read_write> u [[texture(0)]],
                             texture2d<int, access::read_write> s [[texture(1)]],
                             device uint4 *out [[buffer(0)]],
                             uint t [[thread_position_in_grid]]) {
    switch (t >> 31) {
        uint4 stale;
        uint4 current;
    default:
        uint2 xy = uint2(t, 0);
        out[10 * t] = u.atomic_fetch_add(xy, uint4(1u, 2u, 3u, 4u));
        out[10 * t + 1] = u.atomic_fetch_or(xy, uint4(0x100u, 0x200u, 0x300u, 0x400u));
        out[10 * t + 2] = u.atomic_exchange(xy, uint4(0x50u + t, 7u, 8u, 9u));
        out[10 * t + 3] = u.atomic_load(xy);
        stale = uint4(0xE0u, 0xE1u, 0xE2u, 0xE3u);
        bool failed = u.atomic_compare_exchange_weak(xy, &stale, uint4(0x70u, 1u, 2u, 3u));
        out[10 * t + 4] = stale;
        current = uint4(0x50u + t, 0xF1u, 0xF2u, 0xF3u);
        bool swapped = u.atomic_compare_exchange_weak(xy, &current, uint4(0x60u + t, 1u, 2u, 3u));
        out[10 * t + 5] = current;
        out[10 * t + 6] = uint4(uint(failed), uint(swapped), 0u, 0u);
        out[10 * t + 7] = as_type<uint4>(s.atomic_fetch_min(xy, int4(-5, 6, 7, 8)));
        u.atomic_store(uint4(0x90u + t, 0xA1u, 0xA2u, 0xA3u), xy);
        out[10 * t + 8] = u.atomic_load(xy);
        out[10 * t + 9] = as_type<uint4>(s.atomic_exchange(xy, int4(-7, 1, 2, 3)));
    }
}
