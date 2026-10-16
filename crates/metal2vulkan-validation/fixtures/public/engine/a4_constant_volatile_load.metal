// Owned fixture: a volatile load of a program-scope `constant` array element
// at a constant index. The compiler folds an ordinary load of a constant initializer, so only a
// volatile one reaches the AIR as a load of the closed constant object (`access.constant.load`;
// a load of a read-only typed constant object carries no `Volatile`). Each thread writes
// out[t] = table[1] + t = (5 + t, 6 + t, 7 + t, 8 + t). Loop-free.
#include <metal_stdlib>
using namespace metal;

constant uint4 table[2] = { uint4(1, 2, 3, 4), uint4(5, 6, 7, 8) };

kernel void a4_constant_volatile_load(device uint4 *out [[buffer(0)]],
                                      uint t [[thread_position_in_grid]]) {
    out[t] = *(constant volatile uint4 *)&table[1] + uint4(t);
}
