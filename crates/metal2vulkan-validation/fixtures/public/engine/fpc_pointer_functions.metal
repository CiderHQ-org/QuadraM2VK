// Owned MSL: visible functions that `fpc_chosen_pointers`'s
// kernels call through pointers chosen at run time. Three have the type `uint(uint)` a pointer
// holds; `fpc_rec_even` and `fpc_rec_odd` recurse through a pointer each chooses between them,
// `n` of `fpc_rec_even` and `2 n` of `fpc_rec_odd` per call, to depth `n`. The kernel only gives
// the module an entry point. Every recursion ends at its input's depth.
#include <metal_stdlib>
using namespace metal;

using op_t = uint(uint);

[[visible]] uint fpc_inc(uint x) { return x + 1u; }

[[visible]] uint fpc_dbl(uint x) { return x * 2u; }

[[visible]] uint fpc_sq(uint x) { return x * x; }

[[visible]] uint fpc_rec_odd(uint n);

[[visible]] uint fpc_rec_even(uint n) {
    if (n == 0u) {
        return 0u;
    }
    op_t *next = (n & 2u) != 0u ? fpc_rec_odd : fpc_rec_even;
    return n + next(n - 1u);
}

[[visible]] uint fpc_rec_odd(uint n) {
    if (n == 0u) {
        return 0u;
    }
    op_t *next = (n & 2u) != 0u ? fpc_rec_even : fpc_rec_odd;
    return 2u * n + next(n - 1u);
}

kernel void fpc_pointer_functions(device uint *out [[buffer(0)]],
                                  uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
