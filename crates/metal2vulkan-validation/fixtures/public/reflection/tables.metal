// Owned fixture: a visible function table argument, whose
// reflection Metal answers as MTLBindingTypeVisibleFunctionTable, and the visible function it
// calls. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] int reflection_visible(int x) { return x + 1; }

kernel void reflection_tables(visible_function_table<int(int)> table [[buffer(0)]],
                              device int *out [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
    out[tid] = table[0](int(tid));
}
