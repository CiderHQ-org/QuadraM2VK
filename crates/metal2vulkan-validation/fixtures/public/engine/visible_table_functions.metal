// Owned MSL: visible functions a pipeline places in a visible-function table.
// Three have the table's type `uint(uint, uint)` and one another type. The kernel only gives the
// module an entry point. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] uint table_add(uint a, uint b) { return a + b; }

[[visible]] uint table_mul(uint a, uint b) { return a * b; }

[[visible]] uint table_sub(uint a, uint b) { return a - b; }

[[visible]] float table_half(float a) { return a * 0.5f; }

kernel void visible_table_functions(device uint *out [[buffer(0)]],
                                    uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
