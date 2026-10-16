// Owned fixture: loads and stores past a sized thread-local array, the sized
// array containment form. Each thread's `Frame` holds the four-word array between four words
// before it and eight words past it, so every index the case passes stays inside the frame (the
// thread object Metal allocates) while it reaches past the array. Eight threads; thread t
// initializes its frame, stores 0xC0000000 | t to array[s] with s = int(idx[8 + t]) (a signed
// index, so it may lie before the array), then loads array[l] with l = int(idx[t]). out[8t] is
// that load, out[8t + 1 .. 8t + 4] the array's four words after the store, out[8t + 5] the bits
// the store changed in the words before the array, out[8t + 6] those it changed in the words past
// it, and out[8t + 7] is 0xF0000000 | t. Loop-free.
//
// The frame is declared ahead of the switch's only label, so every path into the switch bypasses
// its declaration and Clang emits no lifetime markers for it: LLVM 22 and later read a marker in a
// one-argument form, so a fixture holding one would decode to a different declaration under each
// `llvm-as` a gate assembles it with. The switch is on t >> 31, zero for every thread.
#include <metal_stdlib>
using namespace metal;

struct Frame {
    uint before[4];
    uint array[4];
    uint after[8];
};

kernel void ob2_sized_array(device const uint *idx [[buffer(0)]],
                            device uint *out [[buffer(1)]],
                            uint t [[thread_position_in_grid]]) {
    switch (t >> 31) {
        Frame f;
    default:
        f.before[0] = 0xB0000000u; f.before[1] = 0xB0000001u;
        f.before[2] = 0xB0000002u; f.before[3] = 0xB0000003u;
        f.array[0] = 0xA0000000u; f.array[1] = 0xA0000001u;
        f.array[2] = 0xA0000002u; f.array[3] = 0xA0000003u;
        f.after[0] = 0xD0000000u; f.after[1] = 0xD0000001u;
        f.after[2] = 0xD0000002u; f.after[3] = 0xD0000003u;
        f.after[4] = 0xD0000004u; f.after[5] = 0xD0000005u;
        f.after[6] = 0xD0000006u; f.after[7] = 0xD0000007u;
        f.array[int(idx[8 + t])] = 0xC0000000u | t;
        uint loaded = f.array[int(idx[t])];
        uint before = (f.before[0] ^ 0xB0000000u) | (f.before[1] ^ 0xB0000001u) |
                      (f.before[2] ^ 0xB0000002u) | (f.before[3] ^ 0xB0000003u);
        uint after = (f.after[0] ^ 0xD0000000u) | (f.after[1] ^ 0xD0000001u) |
                     (f.after[2] ^ 0xD0000002u) | (f.after[3] ^ 0xD0000003u) |
                     (f.after[4] ^ 0xD0000004u) | (f.after[5] ^ 0xD0000005u) |
                     (f.after[6] ^ 0xD0000006u) | (f.after[7] ^ 0xD0000007u);
        device uint *o = out + 8 * t;
        o[0] = loaded;
        o[1] = f.array[0];
        o[2] = f.array[1];
        o[3] = f.array[2];
        o[4] = f.array[3];
        o[5] = before;
        o[6] = after;
        o[7] = 0xF0000000u | t;
    }
}
