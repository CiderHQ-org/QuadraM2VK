// Owned MSL: kernels that call visible functions through
// `visible_function_table`s held in argument buffers. The functions come from
// `visible_table_functions.metal`, which the pipeline links; the tables' entries are its function
// handles. Each kernel writes, per thread, entry `i % size` of its table called with (tid, k), or
// 0xfffffffd for a null entry, 0xfffffffe for an empty table, and 0xffffffff for a null table.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

using WordOp = uint(uint, uint);

static uint call_entry(visible_function_table<WordOp> ops, uint i, uint tid, uint k) {
    if (is_null_visible_function_table(ops)) {
        return 0xffffffffu;
    }
    uint n = ops.size();
    if (n == 0u) {
        return 0xfffffffeu;
    }
    WordOp *f = ops[i % n];
    return f == nullptr ? 0xfffffffdu : f(tid, k);
}

// One table member.
struct OneTable {
    visible_function_table<WordOp> ops;
};

kernel void argument_table_dispatch(device uint *out [[buffer(0)]],
                                    device const OneTable &args [[buffer(1)]],
                                    uint tid [[thread_position_in_grid]]) {
    out[tid] = call_entry(args.ops, tid, tid, 3u);
}

// A member array of two tables after a word: thread `tid` calls through table `tid & 1`.
struct TwoTables {
    uint pad;
    visible_function_table<WordOp> tables[2];
};

kernel void argument_table_array_dispatch(device uint *out [[buffer(0)]],
                                          device const TwoTables &args [[buffer(1)]],
                                          uint tid [[thread_position_in_grid]]) {
    out[tid] = call_entry(args.tables[tid & 1u], tid >> 1, tid, 5u);
}

// Two table members, one picked per thread: a merge of two loaded tables.
struct PairTables {
    visible_function_table<WordOp> a;
    visible_function_table<WordOp> b;
};

kernel void argument_table_select(device uint *out [[buffer(0)]],
                                  device const PairTables &args [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]]) {
    visible_function_table<WordOp> ops = (tid & 1u) != 0u ? args.a : args.b;
    out[tid] = call_entry(ops, tid >> 1, tid, 7u);
}
