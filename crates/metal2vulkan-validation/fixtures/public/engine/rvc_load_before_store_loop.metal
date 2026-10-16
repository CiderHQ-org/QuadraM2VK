// Owned fixture (written for this repository): a word loaded
// from a device buffer before a loop, read once in each iteration, while the loop stores to the
// element it was loaded from. Every iteration reads the value loaded before the loop, whatever
// the loop has stored there since. MoltenVK 1.4.1's SPIRV-Cross writes such a load into the text
// of the instruction that reads it, so without the workaround the loop reads that element again
// on each iteration. The value is read once per iteration and used whole (a second read, or a
// value built from it, would make SPIRV-Cross declare it before the loop on its own). The slots,
// the loaded element and the trip count come from buffers, so the compiler can neither forward a
// store to the load, keep the stores in registers, nor unroll the loop. One thread. Bounded: one
// loop of `count` iterations, `count` from a buffer the case fills.
#include <metal_stdlib>
using namespace metal;

kernel void load_before_a_loop_that_stores_over_it(device uint *values [[buffer(0)]],
                                                   const device uint *slots [[buffer(1)]],
                                                   constant uint &count [[buffer(2)]])
{
    const uint v = values[slots[count]];
    for (uint i = 0; i < count; ++i) {
        values[slots[i]] = v * (i + 2);
    }
}
