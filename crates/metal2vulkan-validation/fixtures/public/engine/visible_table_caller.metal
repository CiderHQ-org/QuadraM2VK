// Owned MSL: kernels that call visible functions through a
// `visible_function_table` and through function pointers. The functions come from
// `visible_table_functions.metal`, which the pipeline links; the table's entries are its
// function handles. Loop-free.
#include <metal_stdlib>
using namespace metal;

using WordOp = uint(uint, uint);

[[visible]] uint table_add(uint a, uint b);
[[visible]] uint table_mul(uint a, uint b);

// Calls entry `tid % size` of `ops` with (tid, 3); a null entry writes 0xfffffffd, an empty table
// 0xfffffffe, and a null table 0xffffffff.
kernel void visible_table_dispatch(device uint *out [[buffer(0)]],
                                   visible_function_table<WordOp> ops [[buffer(1)]],
                                   uint tid [[thread_position_in_grid]]) {
    uint r = 0xffffffffu;
    if (!is_null_visible_function_table(ops)) {
        uint n = ops.size();
        r = 0xfffffffeu;
        if (n > 0u) {
            WordOp *f = ops[tid % n];
            r = f == nullptr ? 0xfffffffdu : f(tid, 3u);
        }
    }
    out[tid] = r;
}

// A helper that calls the function it is given: once inlined, each call's callee is a visible
// function's constant address.
__attribute__((noinline)) static uint apply(WordOp *f, uint a, uint b) { return f(a, b); }

kernel void visible_pointer_dispatch(device uint *out [[buffer(0)]],
                                     uint tid [[thread_position_in_grid]]) {
    out[tid] = apply(table_add, tid, 5u) + apply(table_mul, tid, 2u);
}
