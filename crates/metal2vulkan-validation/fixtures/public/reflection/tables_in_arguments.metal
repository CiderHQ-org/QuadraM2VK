// Owned fixture: visible function tables inside an
// argument buffer, one and an array of two beside an inline constant, whose reflection no earlier
// probe measured, and the visible function they call. Written for this repository; no
// third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] uint reflection_table_step(uint x) { return x + 1u; }

struct ReflectionTables {
    visible_function_table<uint(uint)> one [[id(0)]];
    uint scale [[id(1)]];
    array<visible_function_table<uint(uint)>, 2> two [[id(2)]];
};

kernel void reflection_tables_in_arguments(constant ReflectionTables &args [[buffer(0)]],
                                           device uint *out [[buffer(1)]],
                                           uint tid [[thread_position_in_grid]]) {
    out[tid] = args.one[0](tid) * args.scale + args.two[1][0](tid);
}
