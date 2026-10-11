// Owned fixture (milestone M1): the smallest kernel the engine translates end to end.
// Every invocation stores the same constant to word 0 of buffer 0. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void store_const(device uint *out [[buffer(0)]]) {
    out[0] = 42u;
}
